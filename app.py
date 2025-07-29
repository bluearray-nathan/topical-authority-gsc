import os, json
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

# ── 1. Auto‑load credentials ────────────────────────────────────
SA_PATH = "keyword-embeddings-fcfb926ab1c3.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_PATH
sa = json.load(open(SA_PATH))
PROJECT_ID = sa["project_id"]
REGION     = "us-central1"

# hard‑coded OpenAI key
os.environ["OPENAI_API_KEY"] = (
    "sk-proj-RNfKErSh2yZuiyoZG7vHUtKFxdp2id5mwUmd2fUX6Bq25w-"
    "MfHxCAia-IqoPYR1idH_Kluv43ET3BlbkFJ-zMJeB__l6m0hDi6X7n37yt"
    "Ls6JjzJWfqWjeArZh6hd4yGMqM3ydV2MtUZIbkfIQzERr3CHRQA"
)

# ── 2. Sidebar controls ─────────────────────────────────────────
st.sidebar.header("🔧 Clustering Settings")
min_cluster_size      = st.sidebar.slider("HDBSCAN min_cluster_size", 2, 100, 15)
min_samples           = st.sidebar.slider("HDBSCAN min_samples", 1, 10, 2)
cluster_eps           = st.sidebar.slider("cluster_selection_epsilon", 0.0, 1.0, 0.1)
post_assign_threshold = st.sidebar.slider("Noise→Cluster sim thresh", 0.5, 0.9, 0.7)

# ── 3. Init clients ─────────────────────────────────────────────
client       = OpenAI()
aiplatform.init(project=PROJECT_ID, location=REGION)
genai_client = genai.Client()

# ── 4. App UI & pipeline ────────────────────────────────────────
st.title("🔍 GSC‑to‑Topics Pipeline")

uploaded = st.file_uploader("Upload your GSC CSV", type="csv")
if not uploaded:
    st.info("Please upload a Google Search Console CSV to continue.")
    st.stop()

df = pd.read_csv(uploaded)
st.write("Raw data sample:", df.head())

# Clean & normalize
df = df.rename(columns={
    "Query":"query","Clicks":"clicks",
    "Impressions":"impressions","CTR":"ctr","Position":"avg_pos"
})
df["ctr"] = pd.to_numeric(df["ctr"].astype(str).str.rstrip("%"), errors="coerce").fillna(0)
for c in ["clicks","impressions","avg_pos"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# Embedding function
@st.cache_data(show_spinner=False)
def get_embeddings(texts):
    out = []
    for t in tqdm(texts, desc="Embedding…", leave=False):
        resp = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=t,
            config=EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=3072
            )
        )
        out.append(resp.embeddings[0].values)
    return np.array(out)

embs = get_embeddings(df["query"].tolist())

# PCA + HDBSCAN
coords = PCA(n_components=20, random_state=42).fit_transform(embs)
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    cluster_selection_epsilon=cluster_eps,
    metric="euclidean"
)
df["cluster"] = clusterer.fit_predict(coords)
st.write(f"Clusters found: {len(set(df['cluster'])) - ( -1 in df['cluster'])}, noise: {(df['cluster']==-1).sum()}")

# Post‑assign noise
centroids = {
    cid: coords[idxs].mean(axis=0)
    for cid, idxs in df[df.cluster!=-1].groupby("cluster").indices.items()
}
def reassign(i):
    if df.at[i,"cluster"] != -1:
        return df.at[i,"cluster"]
    vec = coords[i].reshape(1,-1)
    sims = {cid: cosine_similarity(vec, centroids[cid].reshape(1,-1))[0][0] for cid in centroids}
    best,score = max(sims.items(), key=lambda x: x[1])
    return best if score>=post_assign_threshold else -1

df["cluster"] = [reassign(i) for i in df.index]
st.write(f"After post‑assign noise: {(df['cluster']==-1).sum()}")

# GPT labeling
@st.cache_data(show_spinner=False)
def label_clusters(clusts):
    mapping = {}
    for cid in sorted(set(clusts)):
        if cid==-1: continue
        sample = df[df.cluster==cid]["query"].tolist()[:10]
        prompt = "Here are some queries:\n" + "\n".join(f"- {q}" for q in sample)
        prompt += "\n\nProvide a concise topic name (3 words max):"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=20
        )
        mapping[cid] = resp.choices[0].message.content.strip()
    return mapping

labels = label_clusters(df["cluster"])
df["topic"] = df["cluster"].map(lambda c: labels.get(c, "Noise"))

st.write("Sample topics:", df[["query","topic"]].head())

# Download
csv = df.to_csv(index=False)
st.download_button("Download keywords_with_topics.csv", csv, "keywords_with_topics.csv", "text/csv")
