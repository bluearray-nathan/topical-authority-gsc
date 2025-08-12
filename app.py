# app.py
# Topical Hubs (LLM IA, clusters-only) with Map-Reduce taxonomy, consolidation, and gap-fill.
# - Builds a taxonomy (hubs) from ALL clusters via shards, consolidates, then assigns every cluster.
# - Preserves exact CSV headers & order. Overwrites only the values in "Topical cluster".
#
# Requirements:
# - Streamlit secrets configured for Vertex + OpenAI:
#   st.secrets["vertex_ai"] (service account dict), st.secrets["openai"]["api_key"]
# - CSV must contain: Keyword, Search volume, Cluster, Topical cluster (others preserved)

import os
import io
import json
import time
import hashlib
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize
from openai import OpenAI
from google.cloud import aiplatform
from google import genai
from google.genai.types import EmbedContentConfig

# ──────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Topical Hubs (LLM IA, clusters-only)", page_icon="🧭", layout="wide")
st.title("🧭 Build Topical Hubs (LLM IA, clusters-only)")
st.caption("Generates a hub taxonomy from your clusters (map-reduce) and assigns each cluster. Only 'Topical cluster' values are changed; headers/order preserved.")

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

# Map-Reduce taxonomy controls
use_map_reduce       = st.sidebar.checkbox("Use map-reduce taxonomy (cover all clusters)", value=True)
shard_size           = st.sidebar.slider("Shard size (clusters per shard)", 600, 1500, 1000, 50)
hubs_per_shard       = st.sidebar.slider("Max hubs per shard", 20, 80, 50, 5)
consolidation_sim    = st.sidebar.slider("Consolidation similarity (hub title+accepts)", 0.86, 0.98, 0.91, 0.01)
gapfill_max_new_hubs = st.sidebar.slider("Gap-fill: max extra hubs", 10, 100, 40, 5)

# Legacy single-shot sampling (optional)
sample_top_n = st.sidebar.slider("Clusters used to design taxonomy (legacy single-shot)", 300, 2000, 1000, 50)

# Max hubs & regen/debug controls
max_hubs = st.sidebar.slider("Max hubs in final taxonomy", 60, 250, 150, 10)
regenerate_taxonomy = st.sidebar.button("🔄 Regenerate taxonomy (ignore cache)")
show_raw_taxonomy   = st.sidebar.checkbox("Show raw taxonomy response (debug)", value=False)

# Strictness via confidence thresholds & similarity guard
strictness = st.sidebar.radio("Assignment strictness", ["Soft", "Medium", "Hard"], index=1)
CONF_THRESH = {"Soft": 0.45, "Medium": 0.60, "Hard": 0.72}
SIM_FLOOR   = {"Soft": 0.78, "Medium": 0.82, "Hard": 0.86}
conf_floor  = CONF_THRESH[strictness]
sim_floor   = SIM_FLOOR[strictness]
use_similarity_guard = st.sidebar.checkbox("Use embedding similarity guard", value=True,
                                           help="Helps prevent obviously wrong matches. Disable if summaries are too minimal.")

# Parallelism & retries
max_workers_assign = st.sidebar.slider("Parallel assignment workers", 1, 96, 48, 1)
rate_limit_delay   = st.sidebar.slider("Base backoff on 429/5xx (seconds)", 0.5, 10.0, 2.0, 0.5)
temperature        = st.sidebar.slider("GPT temperature (IA & assignment)", 0.0, 1.0, 0.2, 0.1)

# Summaries & stop-tokens
max_keywords_for_summary = st.sidebar.slider("Keywords per cluster used in summaries", 4, 20, 12, 1)
stop_default = "a level,a-level,level,course,courses,degree,degrees,education,ai,guide,hub,near me,programme,program,study,studies,learn,learning,university,uni,college"
stop_tokens = st.sidebar.text_area(
    "Tokens to downweight/remove in summaries (comma-separated)",
    value=stop_default, height=80
)

# Output policy
keep_identical_headers = st.sidebar.checkbox(
    "Keep output columns identical to input (overwrite only 'Topical cluster')",
    value=True
)

# Reset caches
if st.sidebar.button("🗑️ Reset caches (taxonomy, assignments, embeddings)"):
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    st.success("Caches cleared. Regenerate the taxonomy and re-run assignment.")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: If you see too many 'Misc', lower strictness or increase hubs/shard size, then Regenerate taxonomy.")

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

def call_openai_json(prompt: str, max_tokens: int = 1200, temperature: float = 0.0, retries: int = 4) -> Any:
    """
    Ask for strict JSON and parse; robust to model formatting issues.
    Strategy:
      1) Try with response_format=json_object (supported on 4o/mini).
      2) If parse fails, re-ask with a stricter “ONLY JSON” prompt.
      3) If still failing, run a “JSON repair” pass, then final bracket extraction.
    """
    def _ask(p: str, use_json_mode: bool) -> str:
        kwargs = dict(
            model="gpt-4o-mini",
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": p}],
        )
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = openai_client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    # 1) First try: JSON mode
    try:
        text = _ask(prompt, use_json_mode=True)
        return json.loads(text)
    except Exception:
        pass

    # 2) Retry with stricter wrapper
    strict_prompt = (
        "Return ONLY valid minified JSON. No prose, no markdown, no comments. "
        "If the result is large, reduce the number of items but keep the schema identical.\n\n" + prompt
    )
    for _ in range(max(1, retries - 2)):
        try:
            text = _ask(strict_prompt, use_json_mode=True)
            return json.loads(text)
        except Exception:
            time.sleep(rate_limit_delay)

    # 3) Try to REPAIR malformed JSON
    raw = _ask(prompt, use_json_mode=False)
    try:
        repair_prompt = f"""Fix the following into valid JSON. Output ONLY JSON, no markdown.

-------- START
{raw}
-------- END"""
        fixed = _ask(repair_prompt, use_json_mode=True)
        return json.loads(fixed)
    except Exception:
        # Final fallback: largest {...} or [...] block
        for open_b, close_b in [("{", "}"), ("[", "]")]:
            s, e = raw.find(open_b), raw.rfind(close_b)
            if s != -1 and e != -1 and e > s:
                try:
                    return json.loads(raw[s:e+1])
                except Exception:
                    pass
        raise ValueError("Could not parse JSON from LLM response.")

# ── Consolidation helpers must come AFTER embed_texts_cached ─────────
BANNED_LABELS = ["AI Education Hub", "A Level", "A-Level", "Misc", "General", "Education", "Academics"]

def hub_signature(h: Dict[str, Any]) -> str:
    title = h.get("title", "").strip()
    accepts = ", ".join(h.get("accepts", [])[:12])
    return f"{title} — {accepts}"

# strict JSON caller for large/fragile prompts
def call_openai_json_strict(prompt: str, max_tokens: int, temperature: float, retries: int = 4) -> dict:
    def _ask(p: str, json_mode: bool) -> str:
        kwargs = dict(
            model="gpt-4o-mini",
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": p}],
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return openai_client.chat.completions.create(**kwargs).choices[0].message.content.strip()

    # 1) JSON mode
    try:
        return json.loads(_ask(prompt, True))
    except Exception:
        pass

    # 2) Stricter wrapper
    strict = "Return ONLY valid minified JSON. No prose, no markdown.\n\n" + prompt
    for _ in range(max(1, retries - 2)):
        try:
            return json.loads(_ask(strict, True))
        except Exception:
            time.sleep(rate_limit_delay)

    # 3) Repair then largest-block fallback
    raw = _ask(prompt, False)
    try:
        repair = f"Fix to valid JSON. Output ONLY JSON.\n\n{raw}"
        return json.loads(_ask(repair, True))
    except Exception:
        for ob, cb in [("{","}"), ("[","]")]:
            s, e = raw.find(ob), raw.rfind(cb)
            if s != -1 and e != -1 and e > s:
                try:
                    return json.loads(raw[s:e+1])
                except Exception:
                    pass
        raise ValueError("Could not parse JSON from LLM response.")

def consolidate_hubs_by_similarity(hubs: List[Dict[str, Any]], sim_threshold: float) -> List[Dict[str, Any]]:
    if not hubs:
        return hubs
    texts = [hub_signature(h) for h in hubs]
    V = embed_texts_cached(tuple(texts))
    V = normalize(V, axis=1)
    n = V.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    blk = 1024
    for i0 in range(0, n, blk):
        A = V[i0:i0 + blk]
        for j0 in range(i0, n, blk):
            B = V[j0:j0 + blk]
            S = A @ B.T
            ai, aj = np.where(S >= sim_threshold)
            for k in range(len(ai)):
                i = i0 + ai[k]
                j = j0 + aj[k]
                if i < j:
                    union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    merged = []
    seen_titles = set()
    for root, idxs in groups.items():
        members = [hubs[i] for i in idxs]
        titles = [m.get("title", "").strip() for m in members if m.get("title")]
        best_title = sorted(titles, key=len)[0] if titles else "Topic"
        accepts, avoid = [], []
        for m in members:
            accepts.extend([a for a in m.get("accepts", []) if isinstance(a, str)])
            avoid.extend([a for a in m.get("avoid", []) if isinstance(a, str)])
        accepts = sorted(set(accepts))[:30]
        avoid = sorted(set(avoid))[:30]
        tid = best_title.upper().replace(" ", "_")
        tid = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in tid)
        if best_title.lower() in seen_titles:
            tid = tid + "_" + hashlib.md5(best_title.encode()).hexdigest()[:6].upper()
        seen_titles.add(best_title.lower())
        merged.append({"id": tid, "title": best_title, "parent_id": None, "accepts": accepts, "avoid": avoid})
    return merged

def pick_diverse_hubs(hubs: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    if len(hubs) <= k:
        return hubs
    sigs = [hub_signature(h) for h in hubs]
    V = embed_texts_cached(tuple(sigs))
    V = normalize(V, axis=1)
    # Start with the shortest title (usually cleanest)
    start = int(np.argmin([len(h.get("title", "")) for h in hubs]))
    chosen = [start]
    sims = V @ V[start].reshape(-1, 1)
    min_sim = sims.ravel()
    while len(chosen) < k:
        idx = int(np.argmin(min_sim))
        chosen.append(idx)
        min_sim = np.minimum(min_sim, (V @ V[idx].reshape(-1, 1)).ravel())
    chosen = sorted(set(chosen))
    return [hubs[i] for i in chosen]

# ──────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────
def taxonomy_prompt(sample_items: List[Dict[str, Any]], max_hubs_param: int) -> str:
    return f"""
Design a two-tier site taxonomy (list of hubs) solely from the following cluster summaries.
Respond with strict JSON:
{{
  "hubs": [
    {{
      "id": "UG_BUSINESS",
      "title": "Undergraduate Business Degrees",
      "parent_id": null,
      "accepts": ["business","management","ba","bsc","undergraduate"],
      "avoid":   ["mba","postgraduate","fees"]
    }}
  ],
  "constraints": {{"max_hubs": {max_hubs_param}, "max_depth": 2}}
}}

Rules:
- 60–{max_hubs_param} hubs, 2–4 word Title Case names.
- Avoid banned titles: AI Education Hub, A Level, A-Level, Misc, General, Education, Academics.
- Keep consistent level/mode/geo only if common across items.
- Favor hubs that align to user navigation; keep granularity consistent across hubs.
- IDs must be uppercase (A–Z, 0–9, underscores), unique and stable.
- Include short 'accepts' terms that characterize the hub; include 'avoid' for common confusions.
- Keep depth at most 2 (parent/child). Parent hubs optional.

Cluster summaries (sample):
{json.dumps(sample_items, ensure_ascii=False, indent=2)}

Return JSON only.
""".strip()

def shard_taxonomy_prompt(shard_items: List[Dict[str, Any]], max_hubs_param: int) -> str:
    return f"""
Design a list of up to {max_hubs_param} specific hubs for a university website, from ONLY the following cluster summaries (subset/shard).
Return JSON:
{{"hubs":[{{"id":"...","title":"...","parent_id":null,"accepts":["..."],"avoid":["..."]}}]}}

Rules:
- 2–4 word Title Case titles; avoid: AI Education Hub, A Level, A-Level, Misc, General, Education, Academics.
- Keep consistent level/mode/geo only if common in this shard.
- IDs uppercase underscore, unique within this shard.

Shard cluster summaries:
{json.dumps(shard_items, ensure_ascii=False, indent=2)}

Return JSON only.
""".strip()

def consolidate_taxonomy_prompt(hubs: List[Dict[str, Any]], max_hubs_param: int) -> str:
    return f"""
You are consolidating hub candidates (may contain near-duplicates). Merge synonyms into single hubs.
Return JSON:
{{"hubs":[{{"id":"...","title":"...","parent_id":null,"accepts":["..."],"avoid":["..."]}}]}}

Rules:
- Keep 60–{max_hubs_param} total; prefer the clearest, navigational title.
- Merge close titles (synonyms), union their 'accepts', intersect obvious 'avoid'.
- Avoid banned titles: AI Education Hub, A Level, A-Level, Misc, General, Education, Academics.
- Ensure unique IDs and titles.

Candidates:
{json.dumps(hubs, ensure_ascii=False, indent=2)}

Return JSON only.
""".strip()

def gapfill_taxonomy_prompt(problem_items: List[Dict[str, Any]], max_new: int) -> str:
    return f"""
Propose up to {max_new} NEW hubs to cover the following uncovered/low-confidence clusters.
Return JSON: {{"hubs":[{{"id":"...","title":"...","parent_id":null,"accepts":["..."],"avoid":["..."]}}]}}
Clusters:
{json.dumps(problem_items, ensure_ascii=False, indent=2)}
Return JSON only.
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
# Step 1 — Prepare per-cluster summaries (by Search volume)
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
# Step 2 — Design taxonomy (map-reduce across ALL clusters, or legacy sample)
# ──────────────────────────────────────────────────────────────────────
st.subheader("Step 2 — Designing the taxonomy (hubs)")

if use_map_reduce:
    st.caption("Map-reduce mode: generating per-shard taxonomies across all clusters…")
    all_items = [{"cluster": c, "summary": cluster_summaries[c], "total_search_volume": int(cluster_total_vol[c])}
                 for c in cluster_summaries.keys()]
    all_items_sorted = sorted(all_items, key=lambda x: x["total_search_volume"], reverse=True)
    shards = [all_items_sorted[i:i + shard_size] for i in range(0, len(all_items_sorted), shard_size)]

    bad_shards = []
    per_shard_hubs = []

    # Robust shard runner with char budget + strict JSON
    def run_shard(shard_idx: int, shard: list) -> dict:
        char_budget = 70_000  # keep prompts reasonable
        s_items, chars = [], 0
        for it in shard:
            t = len(it["summary"])
            if chars + t > char_budget:
                break
            s_items.append(it)
            chars += t
        prompt = shard_taxonomy_prompt(s_items, max_hubs_param=hubs_per_shard)
        try:
            return call_openai_json_strict(prompt, max_tokens=1700, temperature=temperature)
        except Exception as e:
            bad_shards.append({"shard_index": shard_idx, "error": str(e), "count": len(s_items)})
            raise

    progress = st.progress(0.0)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(run_shard, i, s): i for i, s in enumerate(shards)}
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            done += 1
            try:
                out = fut.result()
                per_shard_hubs.extend(out.get("hubs", []))
            except Exception:
                st.warning(f"Shard {idx} failed: Could not parse JSON from LLM response.")
            progress.progress(done / len(shards))
    progress.empty()

    if bad_shards:
        bad_df = pd.DataFrame(bad_shards)
        st.download_button("⬇️ Download failed shard report", bad_df.to_csv(index=False),
                           "failed_shards.csv", "text/csv")

    st.caption("Consolidating shard hubs by semantic similarity…")
    consolidated = consolidate_hubs_by_similarity(per_shard_hubs, consolidation_sim)

    st.caption("LLM consolidation of merged hubs (chunked, robust)…")

    # 1) Pre-trim each candidate for compact JSON
    def _trim(h):
        return {
            "id": str(h.get("id", ""))[:80],
            "title": str(h.get("title", ""))[:80],
            "parent_id": h.get("parent_id", None),
            "accepts": [a for a in h.get("accepts", []) if isinstance(a, str)][:12],
            "avoid":   [a for a in h.get("avoid",   []) if isinstance(a, str)][:12],
        }

    candidates = [_trim(h) for h in consolidated]

    # 2) Chunk candidates to avoid oversized prompts
    chunk_size = 280  # ~safe with 1700 tokens
    chunks = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]

    merged_parts = []
    for idx, ch in enumerate(chunks, 1):
        prompt = consolidate_taxonomy_prompt(ch, max_hubs_param=min(max_hubs, hubs_per_shard * 2))
        try:
            out = call_openai_json_strict(prompt, max_tokens=1700, temperature=temperature)
            merged_parts.extend(out.get("hubs", []))
        except Exception:
            st.warning(f"Consolidation chunk {idx} failed; falling back to candidates unchanged for this chunk.")
            merged_parts.extend(ch)

    # 3) Embedding consolidation over all merged parts
    merged_once = consolidate_hubs_by_similarity(merged_parts, consolidation_sim)

    # 4) Final cap to max_hubs:
    if len(merged_once) > max_hubs:
        hubs_list = pick_diverse_hubs(merged_once, max_hubs)
    else:
        hubs_list = merged_once

else:
    # Legacy single-shot sample (optional)
    sorted_clusters = sorted(cluster_total_vol.items(), key=lambda x: x[1], reverse=True)
    sample = [{"cluster": c, "summary": cluster_summaries[c], "total_search_volume": int(v)}
              for c, v in sorted_clusters[:sample_top_n]]

    # Trim if too long
    if len(sample) > 0:
        total_chars = sum(len(s["summary"]) for s in sample)
        if total_chars > 180_000:
            keep = max(400, int(180_000 / (total_chars / len(sample))))
            sample = sample[:keep]
            st.warning(f"Large sample trimmed to {keep} items to ensure valid JSON output.")

    ia_raw = call_openai_json(taxonomy_prompt(sample, max_hubs_param=max_hubs), max_tokens=2000, temperature=temperature)
    hubs_list = ia_raw.get("hubs", [])

# Cleanup & validate hubs_list
seen_ids, seen_titles = set(), set()
cleaned_hubs = []
for h in hubs_list:
    hid = str(h.get("id", "")).upper().strip().replace(" ", "_")
    hid = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in hid)
    title = str(h.get("title", "")).strip()
    if not hid or not title:
        continue
    if hid in seen_ids or title.lower() in seen_titles:
        continue
    if title in {"Misc", "General", "Education", "AI Education Hub", "A Level", "A-Level"}:
        continue
    seen_ids.add(hid); seen_titles.add(title.lower())
    cleaned_hubs.append({
        "id": hid,
        "title": title,
        "parent_id": h.get("parent_id", None),
        "accepts": [a for a in h.get("accepts", [])[:20] if isinstance(a, str)],
        "avoid":   [a for a in h.get("avoid",   [])[:20] if isinstance(a, str)],
    })

if show_raw_taxonomy:
    try:
        st.code(json.dumps({"hubs": hubs_list}, ensure_ascii=False, indent=2)[:4000], language="json")
    except Exception:
        pass

if not cleaned_hubs:
    st.error("No hubs generated. Loosen settings or disable map-reduce temporarily.")
    st.stop()

ia = {"hubs": cleaned_hubs, "constraints": {"max_hubs": max_hubs, "max_depth": 2}}
st.caption("Review/edit the generated taxonomy (optional).")
ia_text = st.text_area("Taxonomy JSON", json.dumps(ia, ensure_ascii=False, indent=2), height=320)
try:
    ia = json.loads(ia_text)
    hubs_list = ia.get("hubs", [])
except Exception as e:
    st.error(f"Invalid JSON in taxonomy editor: {e}")
    st.stop()

if not hubs_list:
    st.error("No hubs after editing.")
    st.stop()

# Build maps for hub metadata & embed for similarity guard
hub_id_to_title  = {h["id"]: h["title"] for h in hubs_list}
hub_id_to_parent = {h["id"]: h.get("parent_id") for h in hubs_list}
hub_summaries = [f'{h["title"]} — ' + ", ".join([t for t in h.get("accepts", []) if isinstance(t, str)]) for h in hubs_list]
hub_ids_ordered = [h["id"] for h in hubs_list]
hub_vecs = normalize(embed_texts_cached(tuple(hub_summaries)), axis=1)

# ──────────────────────────────────────────────────────────────────────
# Step 3 — Assign each cluster to best hub (parallel, cached, similarity guard)
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

def fallback_parent(hid: str) -> str:
    p = hub_id_to_parent.get(hid)
    return p if p else hid

def choose_with_similarity_guard(cluster_vec: np.ndarray, best_id: str, alt_ids: List[str]) -> str:
    if not use_similarity_guard:
        return best_id
    sims: List[Tuple[str, float]] = []
    try:
        i = hub_ids_ordered.index(best_id)
        sims.append((best_id, float((cluster_vec @ hub_vecs[i].reshape(-1, 1)).ravel()[0])))
    except ValueError:
        pass
    for aid in alt_ids[:3]:
        try:
            j = hub_ids_ordered.index(aid)
            sims.append((aid, float((cluster_vec @ hub_vecs[j].reshape(-1, 1)).ravel()[0])))
        except ValueError:
            continue
    sims_sorted = sorted(sims, key=lambda x: (x[0] != best_id, -x[1]))  # prefer best_id then higher sim
    for hid, sim in sims_sorted:
        if sim >= sim_floor:
            return hid
    return best_id

def assign_prompt(cluster_name: str, summary: str, hubs_json: Dict[str, Any], strictness_text: str) -> str:
    return f"""
You assign content clusters to site hubs.

Strictness: {strictness_text}. Choose the best hub for the cluster below.
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

def assign_one(cluster_name: str, summary: str) -> Dict[str, Any]:
    key = assignment_key(cluster_name, summary)
    if key in assign_cache:
        return assign_cache[key]

    prompt = assign_prompt(cluster_name, summary, {"hubs": hubs_list}, strictness)
    result = call_openai_json(prompt, max_tokens=300, temperature=temperature)

    best = result.get("best", "")
    confidence = float(result.get("confidence", 0.0) or 0.0)
    alts = result.get("alts", [])
    alt_ids = [a[0] for a in alts if isinstance(a, list) and a and isinstance(a[0], str)]

    chosen = best
    if confidence < conf_floor:
        chosen = fallback_parent(best)

    try:
        cvec = embed_cluster_summary(summary)
        chosen = choose_with_similarity_guard(cvec, chosen, alt_ids)
    except Exception:
        pass

    out = {"best": best, "chosen": chosen, "confidence": confidence, "alts": alts}
    assign_cache[key] = out
    return out

cluster_names = list(cluster_summaries.keys())
cluster_summary_list = [cluster_summaries[c] for c in cluster_names]

assigned_map: Dict[str, str] = {}
results_log: List[Tuple[str, Dict[str, Any]]] = []
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
# Step 3b — Gap-fill (optional): propose extra hubs for low-confidence/Misc and reassign
# ──────────────────────────────────────────────────────────────────────
low_conf_or_misc = []
for cname, res in results_log:
    conf = (res or {}).get("confidence", 0.0) if isinstance(res, dict) else 0.0
    chosen_title = assigned_map.get(cname, "Misc")
    if (conf < (conf_floor - 0.1)) or (chosen_title == "Misc"):
        low_conf_or_misc.append({"cluster": cname, "summary": cluster_summaries[cname]})

if low_conf_or_misc and gapfill_max_new_hubs > 0:
    st.caption(f"Gap-fill: proposing up to {gapfill_max_new_hubs} extra hubs for uncovered themes…")
    extra = call_openai_json(
        gapfill_taxonomy_prompt(low_conf_or_misc[:1200], gapfill_max_new_hubs),
        max_tokens=1200, temperature=temperature
    )
    extra_hubs = extra.get("hubs", [])
    if extra_hubs:
        merged = consolidate_hubs_by_similarity(hubs_list + extra_hubs, consolidation_sim)
        # Rebuild maps
        hubs_list = merged
        hub_id_to_title  = {h["id"]: h["title"] for h in hubs_list}
        hub_id_to_parent = {h["id"]: h.get("parent_id") for h in hubs_list}
        hub_summaries = [hub_signature(h) for h in hubs_list]
        hub_ids_ordered = [h["id"] for h in hubs_list]
        hub_vecs = normalize(embed_texts_cached(tuple(hub_summaries)), axis=1)

        def reassign_one(cname):
            res = assign_one(cname, cluster_summaries[cname])
            hid = res.get("chosen") or res.get("best")
            return cname, hub_id_to_title.get(hid, "Misc")

        with ThreadPoolExecutor(max_workers=max_workers_assign) as ex:
            for cname, title in ex.map(reassign_one, [x["cluster"] for x in low_conf_or_misc]):
                assigned_map[cname] = title

# Optional: export assignment issues
issues = []
for cname, res in results_log:
    if isinstance(res, dict):
        if res.get("chosen") is None or hub_id_to_title.get(res.get("chosen", "")) is None:
            issues.append({
                "cluster": cname,
                "reason": "no chosen id or id not in taxonomy",
                "confidence": res.get("confidence"),
                "best": res.get("best"),
                "alts": json.dumps(res.get("alts", []))
            })
    else:
        issues.append({"cluster": cname, "reason": "non-dict result", "raw": str(res)})

if issues:
    issues_df = pd.DataFrame(issues)
    st.download_button("⬇️ Download assignment issues (CSV)", issues_df.to_csv(index=False),
                       file_name="assignment_issues.csv", mime="text/csv")

# ──────────────────────────────────────────────────────────────────────
# Step 4 — Write results (identical headers)
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

# Diagnostics: hub distribution
st.divider()
st.subheader("Hub distribution (top 25)")
dist = pd.Series([assigned_map.get(c, "Misc") for c in cluster_names]).value_counts().head(25).reset_index()
dist.columns = ["Topical cluster", "Clusters"]
st.dataframe(dist, use_container_width=True)































