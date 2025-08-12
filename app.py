# app.py
# LLM IA (clusters-only) → builds a taxonomy (hubs) from your clusters,
# then assigns each cluster to the best hub. Preserves exact CSV headers/order
# and overwrites only the values in "Topical cluster".
#
# Requirements:
# - Streamlit secrets configured for Vertex + OpenAI (same as your prior app)
# - CSV must contain: Keyword, Search volume, Cluster, Topical cluster (others preserved)

import os
import io
import json
import time
import math
import hashlib
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI
from google.cloud import aiplatform
from google import genai
from google.genai.types import EmbedContentConfig

# ──────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Topical Hubs (LLM IA, clusters-only)", page_icon="🧭", layout="wide")
st.title("🧭 Build Topical Hubs (LLM IA, clusters-only)")
st.caption("Generates a clean hub taxonomy from your clusters and assigns each cluster. Only 'Topical cluster' values are changed; all other headers & order preserved.")

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
os.environ["OPENAI_API_KEY"]                = st.secrets["openai"]["api_key"]

aiplatform.init(project=vertex_secret["project_id"], location=region)
genai_client  = genai.Client()
openai_client = OpenAI()

# ──────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Settings")

# How many clusters to sample to design the taxonomy (the hub list)
sample_top_n = st.sidebar.slider("Clusters used to design taxonomy (by volume)", 300, 2000, 1000, 50,
                                 help="A good hub set emerges from ~800–1500 high-signal clusters.")

# Strictness via confidence thresholds for assignment
strictness = st.sidebar.radio(
    "Assignment strictness",
    options=["Soft", "Medium", "Hard"], index=1,
    help="Higher strictness demands stronger matches; fewer forced assignments."
)

CONF_THRESH = {"Soft": 0.45, "Medium": 0.60, "Hard": 0.72}
SIM_FLOOR   = {"Soft": 0.78, "Medium": 0.82, "Hard": 0.86}  # embedding sanity check floor
conf_floor  = CONF_THRESH[strictness]
sim_floor   = SIM_FLOOR[strictness]

# Parallelism & retries
max_workers_assign = st.sidebar.slider("Parallel assignment workers", 1, 96, 48, 1)
rate_limit_delay   = st.sidebar.slider("Base backoff on 429/5xx (seconds)", 0.5, 10.0, 2.0, 0.5)
temperature        = st.sidebar.slider("GPT temperature (naming/IA)", 0.0, 1.0, 0.2, 0.1)

# Summary construction
max_keywords_for_summary = st.sidebar.slider("Keywords per cluster used in summaries", 4, 20, 12, 1)

# Stop tokens (down-weight generic boilerplate; keep geo/level/mode if meaningful)
stop_default = "a level,a-level,level,course,courses,degree,degrees,education,ai,guide,hub,near me,programme,program,study,studies,learn,learning,university,uni,college"
stop_tokens = st.sidebar.text_area(
    "Tokens to downweight/remove in summaries (comma-separated)",
    value=stop_default, height=80
)

# Keep exact headers & order, overwrite only Topical cluster
keep_identical_headers = st.sidebar.checkbox(
    "Keep output columns identical to input (overwrite only 'Topical cluster')",
    value=True
)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Use a small dry run (e.g., sample_top_n=500) to validate the hub set, then run full assignment.")

# ──────────────────────────────────────────────────────────────────────
# Helpers: hashing, caching, embeddings, prompts, retrying
# ──────────────────────────────────────────────────────────────────────
def file_hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

@st.cache_resource(show_spinner=False)
def get_assignment_cache() -> Dict[str, Any]:
    """In-memory cache for assignments: key (cluster summary hash) -> result dict."""
    return {}

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

def build_cluster_summary(name: str, kw_rows: pd.DataFrame, stops: set[str], top_k: int) -> str:
    tmp = kw_rows.copy()
    tmp["__vol__"] = pd.to_numeric(tmp["Search volume"], errors="coerce").fillna(0)
    kws = tmp.sort_values("__vol__", ascending=False)["Keyword"].astype(str).tolist()[:top_k]
    cleaned = [clean_tokens(k, stops) for k in kws]
    cleaned = [c for c in cleaned if c]
    if not cleaned:
        return name
    return f"{name} — " + ", ".join(cleaned)

def call_openai(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    """Robust OpenAI call with backoff; returns raw text."""
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
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            if attempt >= 6:
                raise
            time.sleep(wait)
            wait *= 1.8

def call_openai_json(prompt: str, max_tokens: int = 1200, temperature: float = 0.0) -> Any:
    """Ask for strict JSON and parse; fall back to naive JSON extraction."""
    text = call_openai(prompt, max_tokens=max_tokens, temperature=temperature)
    # Try direct JSON first
    try:
        return json.loads(text)
    except Exception:
        # Naive bracket extraction
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
        # As last resort, try list JSON
        start = text.find("["); end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
        raise ValueError("Could not parse JSON from LLM response.")

# ──────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────
BANNED_LABELS = ["AI Education Hub", "A Level", "A-Level", "Misc", "General", "Education", "Academics"]

def taxonomy_prompt(sample_items: List[Dict[str, Any]], max_hubs: int = 100) -> str:
    """
    Ask GPT to design a taxonomy (hub list) purely from cluster summaries.
    Each hub must have: id, title, accepts[], avoid[], optional parent_id.
    """
    examples = """
Good hub titles (2–4 words):
- Undergraduate Business Degrees
- Online Psychology Courses
- MBA & Executive Management
- Law & Criminology (UK)
- Computing & IT Degrees

Forbidden titles (too generic):
- AI Education Hub
- A Level
- Misc
- Education
- General
""".strip()

    return f"""
Design a two-tier site taxonomy (list of hubs) **solely** from the following cluster summaries.
Respond with strict JSON:
{{
  "hubs": [
    {{
      "id": "UG_BUSINESS",
      "title": "Undergraduate Business Degrees",
      "parent_id": null,
      "accepts": ["business", "management", "ba", "bsc", "undergraduate"],
      "avoid":   ["mba","postgraduate","fees"]
    }}
  ],
  "constraints": {{"max_hubs": {max_hubs}, "max_depth": 2}}
}}

Rules:
- 60–{max_hubs} hubs, 2–4 word **Title Case** names.
- Avoid banned titles: {", ".join(BANNED_LABELS)}.
- Keep consistent level/mode/geo **only if common** across items.
- Favor hubs that align to user navigation; keep granularity consistent.
- IDs must be uppercase snake-like (A–Z, 0–9, underscores), unique and stable.
- Include short 'accepts' terms that characterize the hub; include 'avoid' for common confusions.
- Keep depth at most 2 (parent/child). Parent hubs optional.

Cluster summaries (sample):
{json.dumps(sample_items, ensure_ascii=False, indent=2)}

Return JSON only.
""".strip()

def assign_prompt(cluster_name: str, summary: str, hubs_json: Dict[str, Any], strictness: str) -> str:
    """
    Ask GPT to assign this cluster to best hub: return JSON:
    {"best":"HUB_ID","confidence":0.0,"alts":[["ALT_ID",0.0],...]}
    """
    return f"""
You assign content clusters to site hubs.

Strictness: {strictness}. Choose the best hub for the cluster below.
Return strictly this JSON:
{{"best":"HUB_ID","confidence":0.00,"alts":[["ALT_ID",0.00],["ALT2_ID",0.00]]}}

Rules:
- Prefer hubs whose 'accepts' fit and 'avoid' do not match.
- Keep level/mode/geo if present in the cluster summary.
- Be conservative: if uncertain, lower confidence.
- No commentary.

Hubs JSON:
{json.dumps(hubs_json, ensure_ascii=False)}

Cluster:
- name: {cluster_name}
- summary: {summary}
""".strip()

# ──────────────────────────────────────────────────────────────────────
# Upload CSV
# ──────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload your CSV (must include: Keyword, Search volume, Cluster, Topical cluster)", type=["csv"])
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
# Prepare per-cluster summaries (by Search volume)
# ──────────────────────────────────────────────────────────────────────
st.subheader("Step 1 — Preparing cluster summaries")
stops = set([t.strip().lower() for t in stop_tokens.split(",") if t.strip()])

cluster_groups = {name: grp.copy() for name, grp in df.groupby("Cluster", dropna=False)}

cluster_summaries: Dict[str, str] = {}
cluster_total_vol: Dict[str, float] = {}

for cname, grp in cluster_groups.items():
    summary = build_cluster_summary(str(cname), grp, stops, max_keywords_for_summary)
    cluster_summaries[str(cname)] = summary
    vol = pd.to_numeric(grp["Search volume"], errors="coerce").fillna(0).sum()
    cluster_total_vol[str(cname)] = float(vol)

# ──────────────────────────────────────────────────────────────────────
# Build taxonomy from a sample of top clusters (by total volume)
# ──────────────────────────────────────────────────────────────────────
st.subheader("Step 2 — Designing the taxonomy (hubs) from clusters")
# Select top-N clusters by total volume
sorted_clusters = sorted(cluster_total_vol.items(), key=lambda x: x[1], reverse=True)
sample = [{"cluster": c, "summary": cluster_summaries[c], "total_search_volume": int(v)}
          for c, v in sorted_clusters[:sample_top_n]]

# Ask GPT for taxonomy JSON (cached by file hash + sample size)
@st.cache_data(show_spinner=False)
def get_taxonomy(file_sig: str, sample_json: str, max_hubs: int) -> Dict[str, Any]:
    prompt = taxonomy_prompt(json.loads(sample_json), max_hubs=max_hubs)
    ia = call_openai_json(prompt, max_tokens=2000, temperature=temperature)
    # Basic validation
    hubs = ia.get("hubs", [])
    # Ensure required fields & uniqueness
    seen_ids, seen_titles = set(), set()
    cleaned_hubs = []
    for h in hubs:
        hid = str(h.get("id", "")).upper().replace(" ", "_")
        title = str(h.get("title", "")).strip()
        if not hid or not title:
            continue
        if hid in seen_ids or title.lower() in seen_titles:
            continue
        seen_ids.add(hid); seen_titles.add(title.lower())
        cleaned_hubs.append({
            "id": hid,
            "title": title,
            "parent_id": h.get("parent_id", None),
            "accepts": h.get("accepts", [])[:20],
            "avoid": h.get("avoid", [])[:20]
        })
    return {"hubs": cleaned_hubs, "constraints": ia.get("constraints", {"max_hubs": max_hubs, "max_depth": 2})}

file_sig = file_hash_bytes(csv_bytes) + f"|sample={sample_top_n}"
ia = get_taxonomy(file_sig, json.dumps(sample, ensure_ascii=False), max_hubs=100)

# Allow user to edit the taxonomy JSON if desired
st.caption("Review/edit the generated taxonomy (optional).")
ia_text = st.text_area("Taxonomy JSON", json.dumps(ia, ensure_ascii=False, indent=2), height=320)
try:
    ia = json.loads(ia_text)
    hubs_list: List[Dict[str, Any]] = ia.get("hubs", [])
except Exception as e:
    st.error(f"Invalid JSON in taxonomy editor: {e}")
    st.stop()

if not hubs_list:
    st.error("No hubs were generated.")
    st.stop()

# Build maps for hub metadata
hub_id_to_title = {h["id"]: h["title"] for h in hubs_list}
hub_id_to_parent = {h["id"]: h.get("parent_id") for h in hubs_list}

# Build hub summaries for embedding sanity checks: title + accepts
hub_summaries = [f'{h["title"]} — ' + ", ".join([t for t in h.get("accepts", []) if isinstance(t, str)]) for h in hubs_list]
hub_ids_ordered = [h["id"] for h in hubs_list]
hub_titles_ordered = [h["title"] for h in hubs_list]

# Pre-embed hubs (once)
st.caption("Embedding hub summaries for assignment sanity checks…")
hub_vecs = embed_texts_cached(tuple(hub_summaries))
hub_vecs = normalize(hub_vecs, axis=1)

# ──────────────────────────────────────────────────────────────────────
# Assign each cluster to best hub (parallel, cached, with embedding check)
# ──────────────────────────────────────────────────────────────────────
st.subheader("Step 3 — Assigning clusters to hubs")

assign_cache = get_assignment_cache()

def assignment_key(cluster_name: str, summary: str) -> str:
    return hashlib.sha256((cluster_name + "||" + summary).encode("utf-8")).hexdigest()

def embed_cluster_summary(summary: str) -> np.ndarray:
    v = embed_texts_cached((summary,))[0]
    v = v.reshape(1, -1)
    v = normalize(v, axis=1)
    return v

def choose_with_similarity_guard(cluster_vec: np.ndarray, best_id: str, alt_ids: List[str]) -> str:
    # Compute cosine sims against hubs for best and alts; if best < floor but an alt passes, switch.
    try:
        idx_best = hub_ids_ordered.index(best_id)
    except ValueError:
        idx_best = None
    sims = []
    if idx_best is not None:
        sims.append((best_id, float((cluster_vec @ hub_vecs[idx_best].reshape(-1,1)).ravel()[0])))
    for aid in alt_ids[:3]:
        try:
            j = hub_ids_ordered.index(aid)
            sims.append((aid, float((cluster_vec @ hub_vecs[j].reshape(-1,1)).ravel()[0])))
        except ValueError:
            continue
    # Pick first with sim >= floor; else keep best_id
    for hid, sim in sorted(sims, key=lambda x: (x[0] != best_id, -x[1])):  # check best first
        if sim >= sim_floor:
            return hid
    return best_id

def fallback_parent(hid: str) -> str:
    # Climb to parent if exists; else keep original
    p = hub_id_to_parent.get(hid)
    return p if p else hid

def assign_one(cluster_name: str, summary: str) -> Dict[str, Any]:
    key = assignment_key(cluster_name, summary)
    if key in assign_cache:
        return assign_cache[key]
    prompt = assign_prompt(cluster_name, summary, {"hubs": hubs_list}, strictness)
    result = call_openai_json(prompt, max_tokens=300, temperature=temperature)
    # Basic shape enforcement
    best = result.get("best", "")
    confidence = float(result.get("confidence", 0.0) or 0.0)
    alts = result.get("alts", [])
    alt_ids = [a[0] for a in alts if isinstance(a, list) and a and isinstance(a[0], str)]
    # Embedding sanity check
    cvec = embed_cluster_summary(summary)
    chosen = best
    if confidence < conf_floor:
        # Try parent fallback
        chosen = fallback_parent(best)
    # Similarity guard (if still low semantic similarity, try an alt that passes)
    chosen = choose_with_similarity_guard(cvec, chosen, alt_ids)
    out = {"best": best, "chosen": chosen, "confidence": confidence, "alts": alts}
    assign_cache[key] = out
    return out

# Build per-cluster summary list
cluster_names = list(cluster_summaries.keys())
cluster_summary_list = [cluster_summaries[c] for c in cluster_names]

# Parallel assignment
assigned_map: Dict[str, str] = {}
progress = st.progress(0.0)
status   = st.empty()

def worker(args):
    cname, summary = args
    try:
        res = assign_one(cname, summary)
        hid = res.get("chosen") or res.get("best")
        title = hub_id_to_title.get(hid, hub_id_to_title.get(res.get("best", ""), "Misc"))
        return cname, title, res
    except Exception as e:
        return cname, "Misc", {"error": str(e)}

tasks = list(zip(cluster_names, cluster_summary_list))
done = 0
results_log = []

with ThreadPoolExecutor(max_workers=max_workers_assign) as ex:
    for cname, title, res in ex.map(worker, tasks):
        assigned_map[cname] = title
        results_log.append((cname, res))
        done += 1
        if done % 50 == 0 or done == len(tasks):
            status.text(f"Assigned {done}/{len(tasks)} clusters")
            progress.progress(done / len(tasks))

progress.empty()
status.empty()

# ──────────────────────────────────────────────────────────────────────
# Apply to DataFrame & download
# ──────────────────────────────────────────────────────────────────────
st.subheader("Step 4 — Write results (identical headers)")
df_out = df.copy()
df_out["Topical cluster"] = df_out["Cluster"].astype(str).map(lambda c: assigned_map.get(c, "Misc"))

# Preserve exact column order
df_out = df_out[original_columns]
csv_out = df_out.to_csv(index=False)

st.download_button(
    "⬇️ Download updated CSV (identical headers)",
    data=csv_out,
    file_name="topical_hubs_llm_clusters_only.csv",
    mime="text/csv"
)
st.success("✅ Done. Only 'Topical cluster' values were updated; headers/order unchanged.")

# ──────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Hub distribution (top 25)")
dist = pd.Series([assigned_map.get(c, "Misc") for c in cluster_names]).value_counts().head(25).reset_index()
dist.columns = ["Topical cluster", "Clusters"]
st.dataframe(dist, use_container_width=True)

st.caption("Tip: If a hub looks too large or too generic, edit the taxonomy JSON above (e.g., split it or refine 'accepts/avoid') and re-run the assignment.")




























