# app.py — Simple, fast hubbing via embeddings + k-NN graph communities
# - No map-reduce prompts, no JSON consolidation.
# - Uses Vertex embeddings for clusters; k-NN graph + similarity threshold → communities.
# - (Optional) GPT only to NAME hubs (small, parallel calls).
# - Output columns are identical to input; only "Topical cluster" is updated.

# ---- Safe watcher so Streamlit starts even on low inotify limits ----
import os
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "poll")
os.environ.setdefault(
    "STREAMLIT_SERVER_FOLDER_WATCH_BLACKLIST",
    '[".venv","venv",".git",".cache","node_modules","/usr","/proc","/home/adminuser/venv"]'
)

import io
import json
import time
import hashlib
from typing import Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from openai import OpenAI
from google.cloud import aiplatform
from google import genai
from google.genai.types import EmbedContentConfig

# ──────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Topical Hubs (Simple Graph)", page_icon="🧭", layout="wide")
st.title("🧭 Build Topical Hubs — Simple Graph Method")
st.caption("Fast: embed clusters → k-NN graph → communities. GPT only names hubs. Output columns unchanged (only 'Topical cluster' is updated).")

# ──────────────────────────────────────────────────────────────────────
# Credentials (Vertex + OpenAI)
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
os.environ["OPENAI_API_KEY"]                = st.secrets["openai"]["api_key"]

aiplatform.init(project=vertex_secret["project_id"], location=region)
genai_client  = genai.Client()
openai_client = OpenAI()

# ──────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Settings")

# Strictness ↔ similarity threshold
strictness = st.sidebar.radio("Clustering strictness", ["Soft", "Medium", "Hard"], index=1,
                              help="Higher threshold = tighter (more) hubs.")
SIM_THRESH = {"Soft": 0.40, "Medium": 0.46, "Hard": 0.53}
sim_threshold = st.sidebar.slider("Similarity threshold", 0.30, 0.70, SIM_THRESH[strictness], 0.01)

k_neighbors = st.sidebar.slider("k-NN neighbors per node (k)", 5, 50, 25, 1,
                                help="Higher k joins more edges; too high may over-merge.")
min_hub_size = st.sidebar.slider("Min hub size (merge tiny groups to nearest hub)", 1, 10, 2, 1)

# Embedding summary options
# Removed slider for "Keywords per cluster used in summaries".
# Hard-code to keep summaries strong and stable.
MAX_KEYWORDS_FOR_SUMMARY = 12
stop_default = "guide,hub,near me"
stop_tokens = st.sidebar.text_area("Downweight/remove tokens in summaries (comma-separated)",
                                   value=stop_default, height=70)
embed_batch_size = st.sidebar.slider("Embedding batch size", 64, 1000, 256, 32)

# GPT naming
use_gpt_names   = st.sidebar.checkbox("Use GPT to name hubs", value=True)
gpt_temperature = st.sidebar.slider("GPT temperature (naming)", 0.0, 1.0, 0.0, 0.1)
naming_workers  = st.sidebar.slider("Parallel naming workers", 1, 64, 16, 1)
timeout_openai  = st.sidebar.slider("OpenAI request timeout (sec)", 10, 180, 60, 10)
rate_limit_delay = st.sidebar.slider("Base backoff on 429/5xx (sec)", 0.5, 10.0, 2.0, 0.5)

# Reset caches
if st.sidebar.button("🗑️ Reset caches (embeddings & naming)"):
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    st.success("Caches cleared.")

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def clean_tokens(s: str, stops: set[str]) -> str:
    txt = str(s).lower().strip()
    parts = [p for p in txt.replace("/", " ").replace("|", " ").split() if p]
    parts = [p for p in parts if p not in stops]
    return " ".join(parts)

def build_cluster_summary(name: str, kw_rows: pd.DataFrame, stops: set[str], top_k: int) -> str:
    tmp = kw_rows.copy()
    tmp["__vol__"] = pd.to_numeric(tmp["Search volume"], errors="coerce").fillna(0)
    kws = tmp.sort_values("__vol__", ascending=False)["Keyword"].astype(str).tolist()[:top_k]
    cleaned = [clean_tokens(k, stops) for k in kws]
    cleaned = [c for c in cleaned if c]
    return f"{name} — " + ", ".join(cleaned) if cleaned else name

@st.cache_data(show_spinner=False)
def embed_texts_cached(texts: Tuple[str, ...]) -> np.ndarray:
    """Vertex embeddings with local cache (embeds each text individually)."""
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

def embed_in_batches(texts: List[str], batch_size: int = 256) -> np.ndarray:
    """Embed in batches with a visible progress bar; uses cached per-batch calls."""
    total = len(texts)
    if total == 0:
        return np.zeros((0, 3072), dtype=np.float32)
    bar = st.progress(0.0)
    chunks = []
    done = 0
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        embs = embed_texts_cached(tuple(batch))  # cached across runs
        chunks.append(embs)
        done += len(batch)
        bar.progress(done / total)
    bar.empty()
    return normalize(np.vstack(chunks), axis=1)

def call_openai_json(prompt: str, max_tokens: int = 200, temperature: float = 0.0, retries: int = 4) -> Any:
    """Strict JSON with retries/repair."""
    def _ask(p: str, json_mode: bool) -> str:
        kwargs = dict(
            model="gpt-4o-mini",
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": p}],
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = openai_client.chat.completions.create(timeout=timeout_openai, **kwargs)
        return (resp.choices[0].message.content or "").strip()

    # 1) JSON mode
    try:
        return json.loads(_ask(prompt, True))
    except Exception:
        pass

    # 2) Strict retry
    strict = "Return ONLY valid minified JSON. No prose, no markdown.\n\n" + prompt
    for _ in range(max(1, retries - 2)):
        try:
            return json.loads(_ask(strict, True))
        except Exception:
            time.sleep(rate_limit_delay)

    # 3) Repair
    raw = _ask(prompt, False)
    try:
        fixed = _ask(f"Fix to valid JSON. Only JSON.\n\n{raw}", True)
        return json.loads(fixed)
    except Exception:
        for ob, cb in [("{","}"), ("[","]")]:
            s, e = raw.find(ob), raw.rfind(cb)
            if s != -1 and e != -1 and e > s:
                try:
                    return json.loads(raw[s:e+1])
                except Exception:
                    pass
        raise ValueError("Could not parse JSON from LLM response.")

# ──────────────────────────────────────────────────────────────────────
# Upload CSV
# ──────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload your CSV (needs: Keyword, Search volume, Cluster, Topical cluster)", type=["csv"])
if not uploaded:
    st.info("Awaiting CSV…")
    st.stop()

csv_bytes = uploaded.getvalue()
df = pd.read_csv(io.BytesIO(csv_bytes))
original_columns = list(df.columns)

required = ["Keyword", "Search volume", "Cluster", "Topical cluster"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required column(s): {', '.join(missing)}")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)
st.write(f"Rows: {len(df):,} | Unique clusters: {df['Cluster'].nunique():,}")

# ──────────────────────────────────────────────────────────────────────
# 1) Build one summary per Cluster & embed (with progress)
# ──────────────────────────────────────────────────────────────────────
st.subheader("Step 1 — Embedding clusters")
stops = set([t.strip().lower() for t in stop_tokens.split(",") if t.strip()])

cluster_groups = {name: grp.copy() for name, grp in df.groupby("Cluster", dropna=False)}
cluster_names = list(map(str, cluster_groups.keys()))

# Use fixed MAX_KEYWORDS_FOR_SUMMARY (slider removed)
summaries = [build_cluster_summary(c, cluster_groups[c], stops, MAX_KEYWORDS_FOR_SUMMARY) for c in cluster_names]

emb = embed_in_batches(summaries, batch_size=embed_batch_size)
st.success(f"✅ Embedded {len(cluster_names):,} clusters.")

# ──────────────────────────────────────────────────────────────────────
# 2) k-NN graph → prune by similarity threshold → connected components
# ──────────────────────────────────────────────────────────────────────
st.subheader("Step 2 — Graph communities (no LLM)")
with st.spinner("Building k-NN graph…"):
    nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(cluster_names)), metric="cosine", n_jobs=-1)
    nn.fit(emb)
    dists, idxs = nn.kneighbors(emb, return_distance=True)
    dists, idxs = dists[:, 1:], idxs[:, 1:]  # drop self
    sims = 1.0 - dists

rows, cols, data = [], [], []
thr = float(sim_threshold)
for i in range(idxs.shape[0]):
    mask = sims[i] >= thr
    if not np.any(mask):
        continue
    js = idxs[i, mask]
    ss = sims[i, mask]
    rows.extend([i] * len(js))
    cols.extend(js.tolist())
    data.extend(ss.tolist())

A = coo_matrix((data, (rows, cols)), shape=(len(cluster_names), len(cluster_names))).tocsr()
A = A.maximum(A.T)  # symmetrize

n_comp, labels = connected_components(A, directed=False)
st.write(f"Initial hubs (components): {n_comp}")

# ──────────────────────────────────────────────────────────────────────
# 3) Merge tiny hubs to nearest neighbor hub (min_hub_size)
# ──────────────────────────────────────────────────────────────────────
st.subheader("Step 3 — Merge tiny hubs (if any)")
label_to_nodes: Dict[int, List[int]] = {}
for i, lab in enumerate(labels):
    label_to_nodes.setdefault(lab, []).append(i)

sizes = {lab: len(nodes) for lab, nodes in label_to_nodes.items()}
small_labels = [lab for lab, sz in sizes.items() if sz < min_hub_size]

if small_labels:
    st.caption(f"Merging {len(small_labels)} tiny hubs (<{min_hub_size}) to nearest large hub…")
    new_labels = labels.copy()
    for lab in small_labels:
        nodes = label_to_nodes[lab]
        for i in nodes:
            row = A.getrow(i)
            if row.nnz == 0:
                # isolated: attach to top kNN neighbor
                j = idxs[i, 0]
                new_labels[i] = new_labels[j]
                continue
            j = row.indices[np.argmax(row.data)]
            new_labels[i] = new_labels[j]
    labels = new_labels
else:
    st.caption("No tiny hubs to merge.")

# Rebuild groups and renumber to 0..H-1
label_to_nodes = {}
for i, lab in enumerate(labels):
    label_to_nodes.setdefault(lab, []).append(i)

unique_labels = sorted(label_to_nodes.keys())
label_map = {lab: k for k, lab in enumerate(unique_labels)}
hub_ids = np.array([label_map[lab] for lab in labels], dtype=int)
n_hubs = len(unique_labels)
st.success(f"✅ Final hubs after merge: {n_hubs}")

# ──────────────────────────────────────────────────────────────────────
# 4) Name hubs (fast). GPT optional; deterministic fallback.
# ──────────────────────────────────────────────────────────────────────
st.subheader("Step 4 — Naming hubs")

# Choose a medoid-like representative by internal edge weight
central_idx = np.zeros(n_hubs, dtype=int)
for lab, nodes in label_to_nodes.items():
    if len(nodes) == 1:
        central_idx[label_map[lab]] = nodes[0]
        continue
    scores = []
    for i in nodes:
        row = A.getrow(i)
        mask = np.isin(row.indices, nodes, assume_unique=False)
        scores.append(row.data[mask].sum() if mask.any() else 0.0)
    best_local = int(np.argmax(scores))
    central_idx[label_map[lab]] = nodes[best_local]

def default_title_from_cluster_name(cname: str) -> str:
    t = " ".join(str(cname).strip().split())
    return t[:80].title() if t else "Topic"

hub_examples: Dict[int, List[str]] = {}
for hid, nodes in label_to_nodes.items():
    hub_examples[label_map[hid]] = [cluster_names[i] for i in nodes[:6]]

def make_prompt(centroid_name: str, centroid_summary: str, examples: List[str]) -> str:
    ex = "\n".join(f"- {e}" for e in examples) if examples else "- (none)"
    return f"""
You are naming a website content hub. Provide a short, navigational, 1–4 word Title Case name.
Use one word if it fully represents the topic; otherwise use up to four words to be as descriptive as needed.

Return ONLY this JSON:
{{"title":"<1-4 word hub name>"}}

Centroid cluster (representative):
- name: {centroid_name}
- summary: {centroid_summary}

Other example clusters in this hub:
{ex}

Rules:
- 1–4 words, Title Case.
- Use one word if it fully describes the hub; otherwise up to four words for clarity.
- Avoid generic names like "Misc", "General", "Education".
- Prefer subject, service, or qualification phrasing that would appear in site navigation.
- No punctuation beyond spaces. No emojis. No quotes.
""".strip()

@st.cache_data(show_spinner=False)
def gpt_name_hub(centroid_name: str, centroid_summary: str, examples: Tuple[str, ...],
                 temperature: float, timeout_s: int) -> str:
    prompt = make_prompt(centroid_name, centroid_summary, list(examples))
    try:
        out = call_openai_json(prompt, max_tokens=60, temperature=temperature)
        title = str(out.get("title", "")).strip()
        return title[:80].title() if title else default_title_from_cluster_name(centroid_name)
    except Exception:
        return default_title_from_cluster_name(centroid_name)

hub_titles: Dict[int, str] = {}
progress = st.progress(0.0)
done = 0

if use_gpt_names:
    tasks = []
    for k in range(n_hubs):
        idx = central_idx[k]
        cname = cluster_names[idx]
        csum  = summaries[idx]
        ex    = tuple(hub_examples.get(k, [])[:5])
        tasks.append((k, cname, csum, ex))

    def worker(task):
        k, cname, csum, ex = task
        title = gpt_name_hub(cname, csum, ex, gpt_temperature, timeout_openai)
        return k, title

    with ThreadPoolExecutor(max_workers=naming_workers) as ex:
        for k, title in ex.map(worker, tasks):
            hub_titles[k] = title
            done += 1
            if done % 10 == 0 or done == n_hubs:
                progress.progress(done / n_hubs)
else:
    for k in range(n_hubs):
        idx = central_idx[k]
        hub_titles[k] = default_title_from_cluster_name(cluster_names[idx])
        done += 1
        if done % 50 == 0 or done == n_hubs:
            progress.progress(done / n_hubs)

progress.empty()
st.success("✅ Hubs named.")

# ──────────────────────────────────────────────────────────────────────
# 5) Write results with identical columns (only 'Topical cluster' changes)
# ──────────────────────────────────────────────────────────────────────
st.subheader("Step 5 — Write results (identical headers)")
cluster_to_title = {cluster_names[i]: hub_titles[hub_ids[i]] for i in range(len(cluster_names))}

df_out = df.copy()
df_out["Topical cluster"] = df_out["Cluster"].astype(str).map(lambda c: cluster_to_title.get(c, "Misc"))

# Preserve exact column order
df_out = df_out[original_columns]
csv_out = df_out.to_csv(index=False)

st.download_button(
    "⬇️ Download updated CSV (identical headers)",
    data=csv_out,
    file_name="topical_hubs_graph.csv",
    mime="text/csv"
)
st.success("✅ Done. Only 'Topical cluster' values were updated; headers/order unchanged.")

# ──────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Diagnostics")
sizes_list = sorted([(hub_titles[label_map[lab]], len(nodes)) for lab, nodes in label_to_nodes.items()],
                    key=lambda x: x[1], reverse=True)
diag = pd.DataFrame(sizes_list, columns=["Topical cluster", "Clusters"])
st.write(f"Hubs: {len(diag):,}")
st.dataframe(diag.head(30), use_container_width=True)



































