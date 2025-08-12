# app.py

import os
import io
import json
import time
import math
import random
import hashlib
import traceback
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from openai import OpenAI
from google.cloud import aiplatform
from google import genai
from google.genai.types import EmbedContentConfig
import hdbscan

# ──────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cluster Renamer + Topical Grouper (Scaled)", page_icon="🧭", layout="wide")
st.title("🧭 Cluster Renamer → Topical Grouper (Scaled)")
st.caption("Keeps your CSV headers identical. Only 'Cluster' and 'Topical cluster' values are updated.")

# ──────────────────────────────────────────────────────────────────────
# Credentials & clients (same pattern you used)
# ──────────────────────────────────────────────────────────────────────
vertex_secret = dict(st.secrets["vertex_ai"])  # copy AttrDict → dict
region        = vertex_secret.get("region", "us-central1")
vertex_secret.pop("region", None)

with open("/tmp/sa.json", "w") as f:
    json.dump(vertex_secret, f)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/sa.json"
os.environ["GOOGLE_CLOUD_PROJECT"]          = vertex_secret["project_id"]
os.environ["GOOGLE_CLOUD_LOCATION"]         = region
os.environ["GOOGLE_GENAI_USE_VERTEXAI"]     = "True"

os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]

aiplatform.init(project=vertex_secret["project_id"], location=region)
genai_client  = genai.Client()
openai_client = OpenAI()

# ──────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Settings")

strictness = st.sidebar.radio(
    "Topical grouping strictness (HDBSCAN)",
    options=["Soft", "Medium", "Hard"],
    index=1,
    help="Soft = broader hubs; Hard = tighter, more specific hubs."
)

# Heuristic presets for HDBSCAN on PCA-reduced cluster-embeddings
HDBSCAN_PRESETS = {
    "Soft":   {"min_cluster_size": 8,  "min_samples": 2},
    "Medium": {"min_cluster_size": 15, "min_samples": 5},
    "Hard":   {"min_cluster_size": 25, "min_samples": 10},
}
hdb_params = HDBSCAN_PRESETS[strictness]

max_keywords_for_naming = st.sidebar.slider(
    "Keywords per cluster sent to GPT (for naming)",
    min_value=5, max_value=30, value=12, step=1
)

temperature = st.sidebar.slider("GPT temperature (naming)", 0.0, 1.0, 0.2, 0.1)
max_workers = st.sidebar.slider("Parallel naming workers", 1, 128, 48, 1,
                                help="Higher = faster until you hit API rate limits. Use with care.")
rate_limit_delay = st.sidebar.slider("Backoff base wait (seconds) on 429/errors", 0.5, 10.0, 2.0, 0.5)

benchmark_mode = st.sidebar.checkbox("Benchmark mode (sample & estimate total time)")
benchmark_sample_size = st.sidebar.slider("Benchmark sample size (# clusters)", 100, 2000, 500, 50)

# ──────────────────────────────────────────────────────────────────────
# Helpers — hashing & caching
# ──────────────────────────────────────────────────────────────────────
def cluster_signature(rows: pd.DataFrame, top_k: int) -> str:
    """Create a stable signature of a cluster content to cache GPT naming."""
    if "Search volume" in rows.columns:
        tmp = rows.copy()
        tmp["__vol__"] = pd.to_numeric(tmp["Search volume"], errors="coerce").fillna(0)
        kws = tmp.sort_values("__vol__", ascending=False)["Keyword"].astype(str).tolist()[:top_k]
    else:
        kws = rows["Keyword"].astype(str).tolist()[:top_k]
    sig = "\n".join(kws)
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()

@st.cache_resource(show_spinner=False)
def get_naming_cache() -> Dict[str, str]:
    """Process-lifetime cache: sig -> GPT name."""
    return {}

# ──────────────────────────────────────────────────────────────────────
# Embedding (Vertex) — cached
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def embed_texts_cached(texts: Tuple[str, ...]) -> np.ndarray:
    """Embed a tuple of texts via Vertex AI and cache locally."""
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
    return np.array(embs, dtype=np.float32)

# ──────────────────────────────────────────────────────────────────────
# GPT prompts
# ──────────────────────────────────────────────────────────────────────
def build_cluster_naming_prompt(cluster_name: str, bullets: str) -> str:
    return f"""You are naming content hubs for an SEO site architecture.

Input: a cluster of closely related keywords (with volumes) about the same intent.
Your job: produce a concise, human-readable hub name suitable for a website category or hub page.

Rules:
- 3–5 words, Title Case.
- No brand names, no brackets, no pipes, no slashes.
- Keep consistent geo terms if they are intrinsic (e.g., “Newcastle”).
- Avoid keyword-stuffing; no “near me” unless intrinsic.
- Prefer natural-sounding names over exact-match keywords.
- Do NOT just repeat the highest-volume keyword if it reads awkwardly.

Existing label (may be messy): {cluster_name}

Keywords:
{bullets}

Return ONLY the name, nothing else.
""".strip()

def build_topical_naming_prompt(member_cluster_names: List[str]) -> str:
    bullets = "\n".join(f"- {n}" for n in member_cluster_names[:20])
    return f"""You are naming a higher-level SEO topic hub based on the following cluster names.

Cluster names:
{bullets}

Rules:
- 2–4 words, Title Case.
- Avoid generic labels like “Services” or “Products”.
- No brand names, brackets, or pipes.
- Keep consistent geo terms if they are intrinsic across the set.

Return ONLY the name, nothing else.
""".strip()

# ──────────────────────────────────────────────────────────────────────
# GPT calling with retries (thread-friendly)
# ──────────────────────────────────────────────────────────────────────
def call_openai_name(prompt: str, temperature: float, max_tokens: int = 20) -> str:
    attempt, wait = 0, rate_limit_delay
    while True:
        attempt += 1
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            name = resp.choices[0].message.content.strip()
            name = " ".join(name.split())
            name = name.replace("|", "").replace("/", " ").strip()
            return name
        except Exception as e:
            if attempt >= 6:
                raise
            time.sleep(wait)
            wait *= 1.8  # exponential backoff

# ──────────────────────────────────────────────────────────────────────
# Naming functions
# ──────────────────────────────────────────────────────────────────────
def gpt_name_cluster(orig_name: str, rows: pd.DataFrame) -> str:
    cache = get_naming_cache()
    sig = cluster_signature(rows, max_keywords_for_naming)
    if sig in cache:
        return cache[sig]

    # build bullets list (top N by volume if available)
    has_vol = "Search volume" in rows.columns
    sample = rows.copy()
    if has_vol:
        sample["__vol__"] = pd.to_numeric(sample["Search volume"], errors="coerce").fillna(0)
        sample = sample.sort_values("__vol__", ascending=False)
    kws = sample["Keyword"].astype(str).tolist()[:max_keywords_for_naming]
    vols = sample["__vol__"].tolist()[:max_keywords_for_naming] if has_vol else [None]*len(kws)

    bullets = "\n".join(
        f"- {k} ({int(v)})" if (has_vol and v is not None) else f"- {k}"
        for k, v in zip(kws, vols)
    )
    prompt = build_cluster_naming_prompt(orig_name, bullets)
    name = call_openai_name(prompt, temperature=temperature, max_tokens=20) or (orig_name or "Unnamed Cluster")
    cache[sig] = name
    return name

def gpt_name_topical(member_names: List[str]) -> str:
    prompt = build_topical_naming_prompt(member_names)
    return call_openai_name(prompt, temperature=temperature, max_tokens=16) or (member_names[0] if member_names else "Topic")

# ──────────────────────────────────────────────────────────────────────
# Parallel naming executor
# ──────────────────────────────────────────────────────────────────────
def parallel_rename_clusters(cluster_groups: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    renamed: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {}
        for orig, grp in cluster_groups.items():
            futs[ex.submit(gpt_name_cluster, str(orig), grp)] = orig

        done_count = 0
        progress = st.progress(0.0)
        status = st.empty()
        total = len(futs)

        for fut in as_completed(futs):
            orig = futs[fut]
            try:
                new_name = fut.result()
            except Exception as e:
                new_name = str(orig) if pd.notna(orig) else "Unnamed Cluster"
                st.warning(f"Naming failed for '{orig}': {e}")
            renamed[orig] = new_name
            done_count += 1
            status.text(f"Renamed {done_count}/{total} clusters")
            progress.progress(done_count / total)

        progress.empty()
        status.empty()
    return renamed

# ──────────────────────────────────────────────────────────────────────
# Summaries & embeddings for topical grouping
# ──────────────────────────────────────────────────────────────────────
def summary_for_group(name: str, grp: pd.DataFrame, top_k: int) -> str:
    # Use top K keywords by volume if present
    if "Search volume" in grp.columns:
        tmp = grp.copy()
        tmp["__vol__"] = pd.to_numeric(tmp["Search volume"], errors="coerce").fillna(0)
        kws = tmp.sort_values("__vol__", ascending=False)["Keyword"].astype(str).tolist()
    else:
        kws = grp["Keyword"].astype(str).tolist()
    kws = kws[:top_k]
    return f"{name} — " + ", ".join(kws)

def embed_cluster_summaries(renamed_groups: Dict[str, pd.DataFrame]) -> Tuple[List[str], np.ndarray]:
    names = list(renamed_groups.keys())
    texts = [summary_for_group(n, renamed_groups[n], max_keywords_for_naming) for n in names]

    emb_bar = st.progress(0.0)
    embeddings = []
    B = 128
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        arr = embed_texts_cached(tuple(batch))
        embeddings.append(arr)
        emb_bar.progress(min(1.0, (i+B)/max(1, len(texts))))
    emb_bar.empty()

    X = np.vstack(embeddings).astype(np.float32)
    return names, X

# ──────────────────────────────────────────────────────────────────────
# Topical grouping with PCA + HDBSCAN
# ──────────────────────────────────────────────────────────────────────
def topical_grouping(names: List[str], X: np.ndarray, preset: Dict[str, int]) -> Dict[str, str]:
    # Normalize (cosine-ish) then reduce for density clustering
    Xn = normalize(X, norm="l2", axis=1)
    n_comp = 50 if Xn.shape[0] > 50 else max(2, min(10, Xn.shape[0]-1))
    Xp = PCA(n_components=n_comp, random_state=42).fit_transform(Xn)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=preset["min_cluster_size"],
        min_samples=preset["min_samples"],
        metric="euclidean",
        cluster_selection_epsilon=0.0,
        cluster_selection_method="eom"
    )
    labels = clusterer.fit_predict(Xp)

    # Build topical groups: label -> member cluster names
    topical_groups = defaultdict(list)
    for name, lab in zip(names, labels):
        topical_groups[int(lab)].append(name)

    # Name each topical group using GPT (based on member cluster names)
    topical_name_map: Dict[int, str] = {}
    for gid, members in topical_groups.items():
        if gid == -1:
            topical_name_map[gid] = "Misc"
        else:
            try:
                topical_name_map[gid] = gpt_name_topical(members)
            except Exception as e:
                st.warning(f"Topical naming failed for group {gid}: {e}")
                topical_name_map[gid] = members[0] if members else f"Group {gid}"

    # Map cluster name -> topical name
    cluster_to_topical = {}
    for gid, members in topical_groups.items():
        for c in members:
            cluster_to_topical[c] = topical_name_map[gid]
    return cluster_to_topical

# ──────────────────────────────────────────────────────────────────────
# Upload CSV
# ──────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload your clustered CSV", type=["csv"])
if not uploaded:
    st.info("Awaiting CSV… It should contain at least: Keyword, Cluster, Topical cluster (case-sensitive).")
    st.stop()

csv_bytes = uploaded.getvalue()
df = pd.read_csv(io.BytesIO(csv_bytes))

st.subheader("Preview")
st.dataframe(df.head(10), use_container_width=True)

# Validate required columns
required_cols = ["Keyword", "Cluster", "Topical cluster"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required column(s): {', '.join(missing)}")
    st.stop()

# Preserve exact column order for final output
original_columns = list(df.columns)

# ──────────────────────────────────────────────────────────────────────
# Optional: Benchmark mode
# ──────────────────────────────────────────────────────────────────────
if benchmark_mode:
    st.subheader("⏱ Benchmark Mode")
    # sample unique clusters
    uniq_clusters = list(df["Cluster"].dropna().astype(str).unique())
    if len(uniq_clusters) == 0:
        st.error("No clusters found to benchmark.")
        st.stop()

    sample_n = min(benchmark_sample_size, len(uniq_clusters))
    sample_clusters = set(random.sample(uniq_clusters, sample_n))
    sample_df = df[df["Cluster"].astype(str).isin(sample_clusters)].copy()

    # Build groups
    sample_groups = {name: grp.copy() for name, grp in sample_df.groupby("Cluster", dropna=False)}

    t0 = time.time()
    sample_renamed = parallel_rename_clusters(sample_groups)
    t1 = time.time()

    # Apply renamed to sample
    sample_df["Cluster"] = sample_df["Cluster"].map(lambda x: sample_renamed.get(x, x))

    # Rebuild groups with renamed
    renamed_groups_sample = {name: grp.copy() for name, grp in sample_df.groupby("Cluster", dropna=False)}

    # Embeddings
    t2 = time.time()
    s_names, sX = embed_cluster_summaries(renamed_groups_sample)
    t3 = time.time()

    # Topical grouping
    t4 = time.time()
    s_map = topical_grouping(s_names, sX, hdb_params)
    t5 = time.time()

    # Stats & projection
    naming_sec   = t1 - t0
    embed_sec    = t3 - t2
    topical_sec  = t5 - t4
    per_cluster  = naming_sec / max(1, len(sample_groups))
    total_unique = df["Cluster"].astype(str).nunique()
    est_total    = per_cluster * total_unique + embed_sec * (total_unique/len(sample_groups)) + topical_sec * (total_unique/len(sample_groups))

    st.info(
        f"Benchmark on {len(sample_groups)} clusters:\n"
        f"- Naming:   {naming_sec:.1f}s total  (~{per_cluster:.2f}s/cluster at concurrency {max_workers})\n"
        f"- Embed:    {embed_sec:.1f}s\n"
        f"- Topical:  {topical_sec:.1f}s\n"
        f"\nProjected rough total for {total_unique} clusters: ~{est_total/60:.1f} minutes "
        f"(very dependent on API latency, rate limits, and concurrency)."
    )
    st.divider()

# ──────────────────────────────────────────────────────────────────────
# 1) Rename clusters (parallel GPT)
# ──────────────────────────────────────────────────────────────────────
st.subheader("Step 1 — Renaming clusters with GPT (parallel)")
cluster_groups = {name: grp.copy() for name, grp in df.groupby("Cluster", dropna=False)}
renamed_map = parallel_rename_clusters(cluster_groups)
df["Cluster"] = df["Cluster"].map(lambda x: renamed_map.get(x, x))

# ──────────────────────────────────────────────────────────────────────
# 2) Embed renamed cluster summaries
# ──────────────────────────────────────────────────────────────────────
st.subheader("Step 2 — Embedding cluster summaries for topical grouping")
renamed_groups = {name: grp.copy() for name, grp in df.groupby("Cluster", dropna=False)}
cluster_names, X = embed_cluster_summaries(renamed_groups)

# ──────────────────────────────────────────────────────────────────────
# 3) HDBSCAN topical grouping (+ GPT labels)
# ──────────────────────────────────────────────────────────────────────
st.subheader(f"Step 3 — Topical grouping with HDBSCAN ({strictness})")
cluster_to_topical = topical_grouping(cluster_names, X, hdb_params)
df["Topical cluster"] = df["Cluster"].map(lambda c: cluster_to_topical.get(c, "Misc"))

# ──────────────────────────────────────────────────────────────────────
# Output — identical columns & order
# ──────────────────────────────────────────────────────────────────────
st.subheader("Result")
st.caption("Only the values in 'Cluster' and 'Topical cluster' have been changed; headers and order are identical.")
st.dataframe(df.head(20), use_container_width=True)

df_out = df[original_columns]
csv_out = df_out.to_csv(index=False)
st.download_button(
    "⬇️ Download updated CSV",
    data=csv_out,
    file_name="clusters_updated.csv",
    mime="text/csv"
)

st.success("✅ Done. Your CSV headers are unchanged.")


























