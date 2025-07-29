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

# ── 1) Load credentials from Streamlit secrets ─────────────────────────
sa = st.secrets["vertex_ai"]
with open("/tmp/sa.json", "w") as f:
    json.dump(sa, f)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/sa.json"
os.environ["GOOGLE_CLOUD_PROJECT"]      = sa["project_id"]
os.environ["GOOGLE_CLOUD_LOCATION"]     = sa.get("region", "us-central1")
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
os.environ["OPENAI_API_KEY"]            = st.secrets["openai"]["api_key"]

# ── 2) Initialize clients ─────────────────────────────────────────────
aiplatform.init(project=sa["project_id"], location=os.environ["GOOGLE_CLOUD_LOCATION"])
genai_client = genai.Client()
client       = OpenAI()

# ── 3) Sidebar: clustering controls ───────────────────────────────────
st.sidebar.header("🔧 Clustering Settings")
min_cluster_size      = st.sidebar.slider("HDBSCAN min_cluster_size", 2, 100, 15)
min_samples           = st.sidebar.slider("HDBSCAN min_samples", 1, 10, 2)
cluster_eps           = st.sidebar.slider("HDBSCAN ε (cluster_selection_epsilon)", 0.0, 1.0, 0.1)
post_assign_threshold = st.sidebar.slider("Noise→Cluster sim threshold", 0.5, 0.9, 0.7)

# ── 4) App title & CSV upload ─────────────────────────────────────────
st.title("🔍 GSC‑to‑Topics Streamlit App")
uploaded = st.file_uploader("1) Upload your GSC CSV", type="csv")
if not uploaded:
    st.info("Please upload a Google Search Console CSV to continue.")
    st.stop()

df = pd.read_csv(uploaded)
st.write("Raw data sample:", df.head())

# ── 5) Clean & normalize ──────────────────────────────────────────────
df = df.rename(columns={
    "Query":       "query",
    "Clicks":      "clicks",
    "Impressions": "impressions",
    "CTR":         "ctr",
    "Position":    "avg_pos"
})
df["ctr"] = (
    df["ctr"]
      .astype(str)
      .str.rstrip("%")
      .pipe(pd.to_numeric, errors="coerce")
      .fillna(0)
)
for col in ["clicks", "impressions", "avg_pos"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# ── 6) Embedding helper ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_embeddings(texts):
    embs = []
    for t in tqdm(texts, desc="Embedding…", leave=False):
        resp = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=t,
            config=EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=3072
            )
        )
        embs.append(resp.embeddings[0].values)
    return np.array(embs)

queries    = df["query"].astype(str).tolist()
embeddings = get_embeddings(queries)

# ── 7) PCA reduction + HDBSCAN clustering ─────────────────────────────
coords = PCA(n_components=20, random_state=42).fit_transform(embeddings)
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    cluster_selection_epsilon=cluster_eps,
    metric="euclidean"
)
df["cluster"] = clusterer.fit_predict(coords)
st.write(f"→ {len(set(df['cluster'])) - (-1 in df['cluster'])} clusters; noise: {(df['cluster']==-1).sum()}")

# ── 8) Post‑assign noise → nearest cluster ────────────────────────────
centroids = {
    cid: coords[idxs].mean(axis=0)
    for cid, idxs in df[df.cluster!=-1].groupby("cluster").indices.items()
}

def reassign(i):
    if df.at[i, "cluster"] != -1:
        return df.at[i, "cluster"]
    vec = coords[i].reshape(1, -1)
    sims = {cid: cosine_similarity(vec, centroids[cid].reshape(1, -1))[0][0]
            for cid in centroids}
    best, score = max(sims.items(), key=lambda x: x[1])
    return best if score >= post_assign_threshold else -1

df["cluster"] = [reassign(i) for i in df.index]
st.write(f"→ After post‑assign noise: {(df['cluster']==-1).sum()}")

# ── 9) GPT‑label first‑layer clusters ────────────────────────────────
@st.cache_data(show_spinner=False)
def label_clusters(clusters):
    mapping = {}
    for cid in sorted(set(clusters)):
        if cid == -1:
            continue
        sample = df[df.cluster==cid]["query"].tolist()[:10]
        prompt = "Here are some search queries:\n" + "\n".join(f"- {q}" for q in sample)
        prompt += "\n\nProvide a concise topic name (3 words max):"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=20
        )
        mapping[cid] = resp.choices[0].message.content.strip()
    return mapping

labels_map = label_clusters(df["cluster"])
df["topic"] = df["cluster"].map(lambda c: labels_map.get(c, "Noise"))

# ── 10) Second‑layer superclustering ─────────────────────────────────
topic_texts = [t for t in df["topic"].unique() if t != "Noise"]
topic_embs  = get_embeddings(topic_texts)
num_topics  = len(topic_embs)
n_comp      = min(5, num_topics-1) if num_topics > 1 else 1
tcoords     = PCA(n_components=n_comp, random_state=42).fit_transform(topic_embs)

super_clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=1, metric="euclidean")
super_ids = super_clusterer.fit_predict(tcoords)
super_map = dict(zip(topic_texts, super_ids))
df["super_id"] = df["topic"].map(lambda t: super_map.get(t, -1))

@st.cache_data(show_spinner=False)
def label_supers(sids):
    mapping = {}
    for sid in sorted(set(sids)):
        members = [t for t, s in super_map.items() if s == sid]
        if sid == -1:
            mapping[sid] = "Misc"
        else:
            prompt = "Here are some topic labels:\n" + "\n".join(f"- {m}" for m in members[:10])
            prompt += "\n\nProvide a broad category name (1-2 words):"
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                max_tokens=10
            )
            mapping[sid] = resp.choices[0].message.content.strip()
    return mapping

super_labels = label_supers(df["super_id"])
df["super_topic"] = df["super_id"].map(lambda x: super_labels.get(x, "Misc"))

# ── 11) Show output & download CSV ───────────────────────────────────
st.write("Output sample:", df[["query","topic","super_topic","clicks","impressions","ctr","avg_pos"]].head())
out_csv = df[["query","topic","super_topic","clicks","impressions","ctr","avg_pos"]].to_csv(index=False)
st.download_button(
    "⬇️ Download keywords_with_topics.csv",
    out_csv,
    "keywords_with_topics.csv",
    "text/csv"
)
