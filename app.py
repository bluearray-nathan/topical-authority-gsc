# app.py

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
from openai import OpenAI
from google.cloud import aiplatform
from google import genai
from google.genai.types import EmbedContentConfig

# ── 1) Load credentials from Streamlit secrets ───────────────────────────
# (Make sure you’ve added these entries into your .streamlit/secrets.toml)
sa = json.loads(st.secrets["vertex_ai"]["service_account"])
with open("/tmp/sa.json", "w") as f:
    f.write(json.dumps(sa))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/sa.json"
os.environ["GOOGLE_CLOUD_PROJECT"]      = sa["project_id"]
os.environ["GOOGLE_CLOUD_LOCATION"]     = st.secrets["vertex_ai"].get("region", "us-central1")
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
# OpenAI key
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]

# ── 2) Initialize clients ────────────────────────────────────────────────
aiplatform.init(project=sa["project_id"], location=os.environ["GOOGLE_CLOUD_LOCATION"])
genai_client = genai.Client()
client       = OpenAI()

# ── 3) Sidebar: clustering controls ─────────────────────────────────────
st.sidebar.header("🔧 Clustering Settings")
min_cluster_size      = st.sidebar.slider("HDBSCAN: min_cluster_size", 2, 100, 15)
min_samples           = st.sidebar.slider("HDBSCAN: min_samples", 1, 10, 2)
cluster_eps           = st.sidebar.slider("cluster_selection_epsilon", 0.0, 1.0, 0.1)
post_assign_threshold = st.sidebar.slider("Noise→Cluster sim thresh", 0.5, 0.9, 0.7)

# ── 4) App title & upload ────────────────────────────────────────────────
st.title("🔍 GSC‑to‑Topics Streamlit App")
uploaded = st.file_uploader("1) Upload your GSC CSV", type="csv")
if not uploaded:
    st.info("Please upload a Google Search Console CSV to proceed.")
    st.stop()

df = pd.read_csv(uploaded)
st.write("Raw data sample:", df.head())

# ── 5) Clean & normalize columns ────────────────────────────────────────
df = df.rename(columns={
    "Query":       "query",
    "Clicks":      "clicks",
    "Impressions": "impressions",
    "CTR":         "ctr",
    "Position":    "avg_pos"
})
df["ctr"] = (df["ctr"].astype(str)
               .str.rstrip("%")
               .pipe(pd.to_numeric, errors="coerce")
               .fillna(0))
for col in ["clicks","impressions","avg_pos"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# ── 6) Embedding function ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_embeddings(texts):
    out = []
    for txt in tqdm(texts, desc="Embedding…", leave=False):
        resp = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=txt,
            config=EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=3072
            )
        )
        out.append(resp.embeddings[0].values)
    return np.array(out)

queries    = df["query"].astype(str).tolist()
embeddings = get_embeddings(queries)

# ── 7) PCA reduction + HDBSCAN clustering ───────────────────────────────
coords = PCA(n_components=20, random_state=42).fit_transform(embeddings)
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    cluster_selection_epsilon=cluster_eps,
    metric="euclidean"
)
df["cluster"] = clusterer.fit_predict(coords)
st.write(f"Found {len({*df['cluster']}) - ( -1 in df['cluster'] ):d} clusters; noise: {(df['cluster']==-1).sum()}")

# ── 8) Post‑assign noise to nearest cluster ──────────────────────────────
centroids = {
    cid: coords[idxs].mean(axis=0)
    for cid, idxs in df[df.cluster!=-1].groupby("cluster").indices.items()
}

def reassign(i):
    if df.at[i,"cluster"] != -1:
        return df.at[i,"cluster"]
    vec = coords[i].reshape(1, -1)
    sims = {cid: cosine_similarity(vec, centroids[cid].reshape(1, -1))[0][0]
            for cid in centroids}
    best, score = max(sims.items(), key=lambda x: x[1])
    return best if score >= post_assign_threshold else -1

df["cluster"] = [reassign(i) for i in df.index]
st.write(f"After post‑assign noise: {(df['cluster']==-1).sum()}")

# ── 9) GPT‑label each cluster ───────────────────────────────────────────
@st.cache_data(show_spinner=False)
def label_clusters(clusts):
    mapping = {}
    for cid in sorted(set(clusts)):
        if cid == -1:
            continue
        sample = df[df.cluster==cid]["query"].tolist()[:10]
        prompt = "Here are some search queries:\n" + "\n".join(f"- {q}" for q in sample)
        prompt += "\n\nProvide a concise topic name (3 words max):"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content": prompt}],
            max_tokens=20
        )
        mapping[cid] = resp.choices[0].message.content.strip()
    return mapping

labels_map = label_clusters(df["cluster"])
df["topic"]    = df["cluster"].map(lambda c: labels_map.get(c,"Noise"))

# ── 10) Second‑layer superclustering ───────────────────────────────────
topic_texts = [t for t in df["topic"].unique() if t!="Noise"]
topic_embs  = get_embeddings(topic_texts)

num_topics  = len(topic_embs)
n_comp      = min(5, num_topics-1) if num_topics>1 else 1
topic_coords = PCA(n_components=n_comp, random_state=42).fit_transform(topic_embs)

super_clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=1, metric="euclidean")
super_ids = super_clusterer.fit_predict(topic_coords)
super_map = dict(zip(topic_texts, super_ids))
df["super_id"] = df["topic"].map(lambda t: super_map.get(t, -1))

@st.cache_data(show_spinner=False)
def label_superclusters(sids):
    smap = {}
    for sid in sorted(set(sids)):
        members = [t for t,s in super_map.items() if s==sid]
        if sid == -1:
            smap[sid] = "Misc"
        else:
            prompt = "Here are some topic labels:\n" + "\n".join(f"- {m}" for m in members[:10])
            prompt += "\n\nProvide a broad category name (1-2 words):"
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content": prompt}],
                max_tokens=10
            )
            smap[sid] = resp.choices[0].message.content.strip()
    return smap

super_labels = label_superclusters(df["super_id"])
df["super_topic"] = df["super_id"].map(lambda x: super_labels.get(x, "Misc"))

# ── 11) Show and download results ─────────────────────────────────────
st.write("Sample output:", df[["query","topic","super_topic","clicks","impressions","ctr","avg_pos"]].head())

csv = df[["query","topic","super_topic","clicks","impressions","ctr","avg_pos"]].to_csv(index=False)
st.download_button(
    label="⬇️ Download keywords_with_topics.csv",
    data=csv,
    file_name="keywords_with_topics.csv",
    mime="text/csv"
)

