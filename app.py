# app.py

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
from openai import OpenAI
from google.cloud import aiplatform
from google import genai
from google.genai.types import EmbedContentConfig

# ── 1) Load credentials from Streamlit secrets ─────────────────────────
vertex = st.secrets["vertex_ai"]
region = vertex.pop("region", "us-central1")
# Write service account JSON to disk
with open("/tmp/sa.json", "w") as f:
    json.dump(vertex, f)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/sa.json"
os.environ["GOOGLE_CLOUD_PROJECT"]      = vertex["project_id"]
os.environ["GOOGLE_CLOUD_LOCATION"]     = region
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
# OpenAI key
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]

# ── 2) Initialize clients ──────────────────────────────────────────────
aiplatform.init(
    project=vertex["project_id"],
    location=region
)
genai_client = genai.Client()
openai_client = OpenAI()

# ── 3) Caching helper for batch embeddings ─────────────────────────────
@st.cache_data(show_spinner=False)
def embed_batch(texts: tuple[str, ...]) -> np.ndarray:
    embs = []
    for txt in texts:
        resp = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=txt,
            config=EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=3072
            )
        )
        embs.append(resp.embeddings[0].values)
    return np.array(embs)

# ── 4) Sidebar controls & progress placeholders ───────────────────────
st.sidebar.header("🔧 Settings")
min_cluster_size      = st.sidebar.slider("HDBSCAN min_cluster_size", 2, 100, 15)
min_samples           = st.sidebar.slider("HDBSCAN min_samples",       1, 10,   2)
cluster_eps           = st.sidebar.slider("ε (cluster_selection_epsilon)", 0.0, 1.0, 0.1)
post_assign_threshold = st.sidebar.slider("Noise→Cluster similarity threshold",
                                          0.0, 1.0, 0.7)
batch_size            = st.sidebar.number_input("Embedding batch size",
                                                min_value=100, max_value=5000,
                                                value=500, step=100)
embed_bar             = st.sidebar.progress(0)
embed_status          = st.sidebar.empty()

# ── 5) Upload CSV ─────────────────────────────────────────────────────
st.title("🔍 GSC‑to‑Topics Streamlit App")
uploaded = st.file_uploader("Upload your Google Search Console CSV", type="csv")
if not uploaded:
    st.info("Awaiting upload…")
    st.stop()

df = pd.read_csv(uploaded)
st.write("Raw data sample:", df.head())

# ── 6) Clean & normalize ──────────────────────────────────────────────
df = df.rename(columns={
    "Query":       "query",
    "Clicks":      "clicks",
    "Impressions": "impressions",
    "CTR":         "ctr",
    "Position":    "avg_pos"
})
df["ctr"] = (
    df["ctr"].astype(str)
           .str.rstrip("%")
           .pipe(pd.to_numeric, errors="coerce")
           .fillna(0)
)
for col in ["clicks", "impressions", "avg_pos"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# ── 7) Batch‑embed with caching & progress ────────────────────────────
queries = df["query"].astype(str).tolist()
total   = len(queries)
all_embs = []

for i in range(0, total, batch_size):
    start, end = i, min(i + batch_size, total)
    embed_status.text(f"Embedding {start+1}–{end} of {total}")
    batch = tuple(queries[start:end])
    batch_embs = embed_batch(batch)       # cached call
    all_embs.extend(batch_embs)
    embed_bar.progress(end / total)

embeddings = np.array(all_embs)
embed_bar.empty()
embed_status.empty()
st.success("✅ Embeddings complete")

# ── 8) PCA reduction ─────────────────────────────────────────────────
with st.spinner("Running PCA…"):
    coords = PCA(n_components=20, random_state=42).fit_transform(embeddings)
st.success("✅ PCA complete")

# ── 9) HDBSCAN clustering ─────────────────────────────────────────────
with st.spinner("Clustering…"):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_eps,
        metric="euclidean"
    )
    df["cluster"] = clusterer.fit_predict(coords)
n_clusters = len(set(df["cluster"])) - (1 if -1 in df["cluster"] else 0)
n_noise    = int((df["cluster"] == -1).sum())
st.success(f"✅ Found {n_clusters} clusters (+{n_noise} noise)")

# ── 10) Post‑assignment of noise via cosine similarity ───────────────
with st.spinner("Re‑assigning noise…"):
    centroids = {
        cid: coords[idxs].mean(axis=0)
        for cid, idxs in df[df.cluster != -1].groupby("cluster").indices.items()
    }

    def reassign(i: int) -> int:
        if df.at[i, "cluster"] != -1:
            return df.at[i, "cluster"]
        vec = coords[i].reshape(1, -1)
        sims = {cid: cosine_similarity(vec,
                                       centroids[cid].reshape(1, -1))[0][0]
                for cid in centroids}
        best, score = max(sims.items(), key=lambda x: x[1])
        return best if score >= post_assign_threshold else -1

    df["cluster"] = [reassign(i) for i in df.index]

st.success("✅ Noise re‑assigned")

# ── 11) GPT‑label first‑layer clusters ────────────────────────────────
cluster_ids = sorted(df["cluster"].unique())
label_bar   = st.progress(0)
labels_map  = {}

for idx, cid in enumerate(cluster_ids):
    if cid == -1:
        labels_map[cid] = "Noise"
    else:
        with st.spinner(f"Labeling cluster {cid}…"):
            sample = df[df.cluster == cid]["query"].tolist()[:10]
            prompt = "Here are some search queries:\n" + "\n".join(f"- {q}" for q in sample)
            prompt += "\n\nProvide a concise topic name (3 words max):"
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20
            )
            labels_map[cid] = resp.choices[0].message.content.strip()
    label_bar.progress((idx + 1) / len(cluster_ids))

df["topic"] = df["cluster"].map(lambda c: labels_map.get(c, "Noise"))
label_bar.empty()
st.success("✅ Topics labeled")

# ── 12) Super‑cluster topics ──────────────────────────────────────────
with st.spinner("Super‑clustering…"):
    topic_texts = [t for t in df["topic"].unique() if t != "Noise"]
    topic_embs  = []
    for t in topic_texts:
        resp = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=t,
            config=EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=3072
            )
        )
        topic_embs.append(resp.embeddings[0].values)

    num_topics = len(topic_embs)
    n_comp     = min(5, num_topics - 1) if num_topics > 1 else 1
    tcoords    = PCA(n_components=n_comp, random_state=42).fit_transform(topic_embs)

    super_clusterer = hdbscan.HDBSCAN(min_cluster_size=3,
                                      min_samples=1,
                                      metric="euclidean")
    super_ids       = super_clusterer.fit_predict(tcoords)
    super_map       = dict(zip(topic_texts, super_ids))
    df["super_id"]  = df["topic"].map(lambda t: super_map.get(t, -1))

st.success("✅ Super‑clustering complete")

# ── 13) Label super‑clusters ─────────────────────────────────────────
super_ids    = sorted(df["super_id"].unique())
super_bar    = st.progress(0)
super_labels = {}

for idx, sid in enumerate(super_ids):
    if sid == -1:
        super_labels[sid] = "Misc"
    else:
        with st.spinner(f"Labeling super‑cluster {sid}…"):
            members = [t for t, s in super_map.items() if s == sid][:10]
            prompt  = "Here are some topics:\n" + "\n".join(f"- {m}" for m in members)
            prompt += "\n\nProvide a broad category name (1-2 words):"
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10
            )
            super_labels[sid] = resp.choices[0].message.content.strip()
    super_bar.progress((idx + 1) / len(super_ids))

df["super_topic"] = df["super_id"].map(lambda x: super_labels.get(x, "Misc"))
super_bar.empty()
st.success("✅ Super‑topics labeled")

# ── 14) Show results & download ───────────────────────────────────────
st.write("Sample output:", df[["query", "topic", "super_topic", "clicks",
                              "impressions", "ctr", "avg_pos"]].head())
csv_data = df[["query", "topic", "super_topic", "clicks",
               "impressions", "ctr", "avg_pos"]].to_csv(index=False)
st.download_button("⬇️ Download CSV", csv_data,
                   "keywords_with_topics.csv", "text/csv")



