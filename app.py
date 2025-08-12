# app.py

import os
import io
import json
import time
import hashlib
from typing import Dict, List, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import hdbscan

from openai import OpenAI
from google.cloud import aiplatform
from google import genai
from google.genai.types import EmbedContentConfig

# ──────────────────────────────────────────────────────────────────────
# Page
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Arden Topical Clusters (Hubs)", page_icon="🧭", layout="wide")
st.title("🧭 Build Topical Clusters (Content Hubs) for Arden-like CSVs")
st.caption("Only the values in 'Cluster' and 'Topical cluster' are changed. All other headers & order are preserved.")

# ──────────────────────────────────────────────────────────────────────
# Credentials (same pattern as before)
# ──────────────────────────────────────────────────────────────────────
vertex_secret = dict(st.secrets["vertex_ai"])
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
    "Grouping strictness (HDBSCAN)",
    options=["Soft", "Medium", "Hard"],
    index=1,
    help="Soft = broader hubs; Hard = tighter, more specific hubs."
)

# Tuned presets: smaller min_cluster_size + epsilon helps avoid mega-buckets
HDBSCAN_PRESETS = {
    "Soft":   {"min_cluster_size": 6,  "min_samples": 1, "epsilon": 0.10},
    "Medium": {"min_cluster_size": 10, "min_samples": 2, "epsilon": 0.06},
    "Hard":   {"min_cluster_size": 16, "min_samples": 4, "epsilon": 0.03},
}
preset = HDBSCAN_PRESETS[strictness]

# After HDBSCAN, merge clusters only if their centroids are very similar
CENTROID_MERGE_THRESH = {"Soft": 0.86, "Medium": 0.89, "Hard": 0.92}
merge_tau = CENTROID_MERGE_THRESH[strictness]

use_pca = st.sidebar.checkbox("Use PCA before HDBSCAN", value=False,
                              help="Skipping PCA often preserves finer distinctions.")
pca_components = st.sidebar.slider("PCA components (if used)", 5, 50, 25, 1)

# Tokens that cause over-merging (downweight/remove in summaries). Keep geo terms.
stop_default = "a level,a-level,level,course,courses,degree,degrees,education,ai,guide,hub,near me,programme,program,study,studies,learn,learning,university,uni,college"
stop_tokens = st.sidebar.text_area(
    "Tokens to downweight/remove (comma-separated)",
    value=stop_default,
    height=80,
    help="Common boilerplate tokens that cause over-merging. Geo tokens are kept."
)

max_keywords_for_summary = st.sidebar.slider("Keywords used in summary per cluster", 4, 20, 10, 1)
max_keywords_for_naming  = st.sidebar.slider("Keywords sent to GPT for cluster naming", 5, 30, 12, 1)

max_workers_topical_naming = st.sidebar.slider("Parallel GPT workers (topical naming)", 1, 64, 16, 1)
temperature = st.sidebar.slider("GPT temperature (naming)", 0.0, 1.0, 0.2, 0.1)
rate_limit_delay = st.sidebar.slider("Backoff base wait on 429/errors (sec)", 0.5, 10.0, 2.0, 0.5)

keep_identical_headers = st.sidebar.checkbox(
    "Keep output columns identical to input (overwrite only 'Cluster' and 'Topical cluster')",
    value=True
)

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def embed_texts_cached(texts: Tuple[str, ...]) -> np.ndarray:
    """Embed texts via Vertex AI and cache locally."""
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

def clean_tokens(s: str, stops: set[str]) -> str:
    txt = s.lower().strip()
    parts = [p for p in txt.replace("/", " ").replace("|", " ").split() if p]
    parts = [p for p in parts if p not in stops]
    return " ".join(parts)

def union_find_merge(centroids: Dict[int, np.ndarray], tau: float) -> Dict[int, int]:
    """Merge clusters whose centroid cosine similarity >= tau."""
    from sklearn.preprocessing import normalize as sknorm
    parent = {i: i for i in centroids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    ids = list(centroids.keys())
    mats = np.vstack([centroids[i] for i in ids])
    mats = sknorm(mats, axis=1)
    sims = mats @ mats.T
    n = sims.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if sims[i, j] >= tau:
                union(ids[i], ids[j])

    for i in ids:
        parent[i] = find(i)
    return parent

def build_cluster_naming_prompt(existing_name: str, bullets: str) -> str:
    return f"""You are naming content hubs for a university website.

Input: a cluster of closely related keywords (with volumes) about the same intent.
Your job: produce a concise, human-readable hub name suitable for a site category or hub page.

Rules:
- 3–5 words, Title Case.
- No brand names, brackets, pipes, or slashes.
- Keep consistent geo or level (e.g., “UK”, “Online”, “Undergraduate”) only if intrinsic.
- Avoid keyword-stuffing; no “near me” unless intrinsic.
- Prefer natural-sounding names over exact-match keywords.
- Do NOT just repeat the highest-volume keyword if it reads awkwardly.

Existing label (may be messy): {existing_name}

Keywords:
{bullets}

Return ONLY the name, nothing else.
""".strip()

def build_topical_name_prompt(member_cluster_names: List[str]) -> str:
    bullets = "\n".join(f"- {n}" for n in member_cluster_names[:24])
    banned = {"AI Education Hub", "A Level", "A-Level", "Misc", "General", "Education", "Academics"}
    examples = """
Good Examples:
- Undergraduate Business Degrees
- Online Psychology Courses
- MBA & Executive Management
- Law & Criminology (UK)
- Computing & IT Degrees

Bad Examples (DO NOT USE):
- AI Education Hub
- A Level
- Misc
- Education
- General
""".strip()
    return f"""Name a single higher-level SEO topic hub based on these cluster names.

Cluster names:
{bullets}

Rules:
- 2–4 words, Title Case.
- Must be specific and navigational (good for a hub page).
- Avoid generic or banned labels: {", ".join(sorted(banned))}.
- Include geo/level (e.g., UK, Online, Undergraduate) only if common across members.
- Return ONLY the name, nothing else.

{examples}
""".strip()

def call_openai_name(prompt: str, temperature: float, max_tokens: int = 24) -> str:
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
        except Exception:
            if attempt >= 6:
                raise
            time.sleep(wait)
            wait *= 1.8

def gpt_name_cluster(orig_name: str, rows: pd.DataFrame, top_k: int) -> str:
    """Return a clean 3–5 word human-readable name for the given cluster using top-K keywords by Search volume."""
    tmp = rows.copy()
    if "Search volume" in tmp.columns:
        tmp["__vol__"] = pd.to_numeric(tmp["Search volume"], errors="coerce").fillna(0)
        tmp = tmp.sort_values("__vol__", ascending=False)
    kws = tmp["Keyword"].astype(str).tolist()[:top_k]
    vols = tmp["__vol__"].tolist()[:top_k] if "__vol__" in tmp.columns else [None]*len(kws)
    bullets = "\n".join(
        f"- {k} ({int(v)})" if (v is not None) else f"- {k}"
        for k, v in zip(kws, vols)
    )
    prompt = build_cluster_naming_prompt(orig_name, bullets)
    name = call_openai_name(prompt, temperature=temperature, max_tokens=20) or (orig_name or "Unnamed Cluster")
    return name

def name_topical_groups(groups: Dict[int, List[str]]) -> Dict[int, str]:
    """Name topical groups from member cluster names (parallel)."""
    out = {}
    with ThreadPoolExecutor(max_workers=max_workers_topical_naming) as ex:
        futs = {}
        for gid, members in groups.items():
            if gid == -1:
                out[gid] = "Misc"
                continue
            prompt = build_topical_name_prompt(members)
            futs[ex.submit(call_openai_name, prompt, temperature, 16)] = gid

        progress = st.progress(0.0)
        done = 0
        total = max(1, len(futs))
        for fut in as_completed(futs):
            gid = futs[fut]
            try:
                nm = fut.result()
                if nm.lower() in {"misc", "general", "education"} or len(nm) < 3:
                    nm = groups.get(gid, ["Topic"])[0]
            except Exception as e:
                nm = groups.get(gid, ["Topic"])[0]
                st.warning(f"Topical naming failed for group {gid}: {e}")
            out[gid] = nm
            done += 1
            progress.progress(done/total)
        progress.empty()
    return out

# ──────────────────────────────────────────────────────────────────────
# Upload CSV (Arden-like schema)
# ──────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload your Arden CSV", type=["csv"])
if not uploaded:
    st.info("Awaiting CSV… Must include at least: Keyword, Search volume, Cluster, Topical cluster.")
    st.stop()

csv_bytes = uploaded.getvalue()
df = pd.read_csv(io.BytesIO(csv_bytes))
st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)

required_cols = ["Keyword", "Search volume", "Cluster", "Topical cluster"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required column(s): {', '.join(missing)}")
    st.stop()

original_columns = list(df.columns)  # preserve exact order
st.write(f"Rows: {len(df):,}  |  Columns: {len(df.columns)}")

# ──────────────────────────────────────────────────────────────────────
# Step 1 — Rename each Cluster with GPT (using member keywords & volumes)
# ──────────────────────────────────────────────────────────────────────
st.subheader("Step 1 — Renaming clusters with GPT")
cluster_groups = {name: grp.copy() for name, grp in df.groupby("Cluster", dropna=False)}
progress = st.progress(0.0)
status   = st.empty()

renamed_map: Dict[str, str] = {}
total = len(cluster_groups)
for i, (orig_name, grp) in enumerate(cluster_groups.items(), start=1):
    status.text(f"Renaming cluster {i}/{total}: {orig_name if isinstance(orig_name, str) else '(blank)'}")
    try:
        new_name = gpt_name_cluster(str(orig_name), grp, max_keywords_for_naming)
    except Exception as e:
        new_name = str(orig_name) if pd.notna(orig_name) else "Unnamed Cluster"
        st.warning(f"Naming failed for '{orig_name}': {e}. Kept original.")
    renamed_map[orig_name] = new_name
    progress.progress(i / total)

progress.empty()
status.empty()
df["Cluster"] = df["Cluster"].map(lambda x: renamed_map.get(x, x))

# ──────────────────────────────────────────────────────────────────────
# Step 2 — Build summaries & embed (one vector per *renamed* cluster)
# Summaries use top member keywords (by volume) from this dataset.
# ──────────────────────────────────────────────────────────────────────
st.subheader("Step 2 — Embedding cluster summaries for topical grouping")

# Re-group by the new Cluster names
renamed_groups = {name: grp.copy() for name, grp in df.groupby("Cluster", dropna=False)}

stops = set([t.strip().lower() for t in stop_tokens.split(",") if t.strip()])

def build_summary_for_group(name: str, grp: pd.DataFrame, top_k: int) -> str:
    tmp = grp.copy()
    tmp["__vol__"] = pd.to_numeric(tmp["Search volume"], errors="coerce").fillna(0)
    kws = tmp.sort_values("__vol__", ascending=False)["Keyword"].astype(str).tolist()[:top_k]
    # clean tokens
    cleaned = []
    for k in kws:
        cleaned.append(clean_tokens(k, stops))
    cleaned = [c for c in cleaned if c]
    if not cleaned:
        return name
    return f"{name} — " + ", ".join(cleaned)

names = list(renamed_groups.keys())
summaries = [build_summary_for_group(n, renamed_groups[n], max_keywords_for_summary) for n in names]

emb_bar = st.progress(0.0)
embeddings = []
B = 128
for i in range(0, len(summaries), B):
    batch = summaries[i:i+B]
    arr = embed_texts_cached(tuple(batch))
    embeddings.append(arr)
    emb_bar.progress(min(1.0, (i+B)/max(1, len(summaries))))
emb_bar.empty()
X = np.vstack(embeddings).astype(np.float32)

# ──────────────────────────────────────────────────────────────────────
# Step 3 — HDBSCAN topical grouping (+ centroid merge)
# ──────────────────────────────────────────────────────────────────────
st.subheader(f"Step 3 — Topical grouping with HDBSCAN ({strictness})")
Xn = normalize(X, norm="l2", axis=1)
Xp = Xn
if use_pca and Xn.shape[0] > pca_components:
    n_comp = min(pca_components, Xn.shape[0]-1)
    Xp = PCA(n_components=max(2, n_comp), random_state=42).fit_transform(Xn)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=preset["min_cluster_size"],
    min_samples=preset["min_samples"],
    metric="euclidean",
    cluster_selection_epsilon=preset["epsilon"],
    cluster_selection_method="eom"
)
labels = clusterer.fit_predict(Xp)

# Collect members per topical label
label_to_names: Dict[int, List[str]] = defaultdict(list)
for nm, lab in zip(names, labels):
    label_to_names[int(lab)].append(nm)

# Centroid-based merge to prevent accidental mega-buckets
st.caption("Applying centroid merge to prevent over-broad buckets…")
centroids: Dict[int, np.ndarray] = {}
for lab, member_names in label_to_names.items():
    idxs = [names.index(nm) for nm in member_names]
    if idxs:
        centroids[lab] = Xn[idxs].mean(axis=0)
if len(centroids) > 1:
    parent_map = union_find_merge(centroids, merge_tau)
    merged_label_to_names: Dict[int, List[str]] = defaultdict(list)
    for lab, member_names in label_to_names.items():
        root = parent_map.get(lab, lab)
        merged_label_to_names[root].extend(member_names)
    # dedupe members per merged group
    label_to_names = {k: sorted(set(v)) for k, v in merged_label_to_names.items()}

# ──────────────────────────────────────────────────────────────────────
# Step 4 — Name topical clusters (from member cluster names only)
# ──────────────────────────────────────────────────────────────────────
st.subheader("Step 4 — Naming topical clusters")
topical_name_map = name_topical_groups(label_to_names)

# Build Cluster -> Topical cluster mapping
cluster_to_topical: Dict[str, str] = {}
for gid, member_names in label_to_names.items():
    tname = topical_name_map.get(gid, "Misc")
    for c in member_names:
        cluster_to_topical[c] = tname

# Apply back to DataFrame, preserving exact headers & order
st.subheader("Result & Download")
df["Topical cluster"] = df["Cluster"].map(lambda c: cluster_to_topical.get(str(c), "Misc"))

# Ensure column order exactly as input
df_out = df[original_columns]
csv_out = df_out.to_csv(index=False)
st.download_button(
    "⬇️ Download updated CSV (identical headers)",
    data=csv_out,
    file_name="arden_topical_clusters_updated.csv",
    mime="text/csv"
)

# Quick distribution preview
st.divider()
st.subheader("Topical distribution (preview)")
preview = (
    pd.Series([cluster_to_topical.get(c, "Misc") for c in names])
      .value_counts().head(25).reset_index()
)
preview.columns = ["Topical cluster", "Clusters"]
st.dataframe(preview, use_container_width=True)



























