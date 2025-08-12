# app.py
# Topical Hubs (LLM IA, clusters-only)
# - Builds a taxonomy (hubs) from your clusters (sampled), then assigns all clusters.
# - Preserves exact CSV headers & order. Overwrites only the values in "Topical cluster".
#
# Requirements:
# - Streamlit secrets configured for Vertex + OpenAI:
#   st.secrets["vertex_ai"] (service account), st.secrets["openai"]["api_key"]
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
st.caption("Generates a clean hub taxonomy from your clusters and assigns each cluster. Only 'Topical cluster' values are changed; all other headers & order preserved.")

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

sample_top_n = st.sidebar.slider(
    "Clusters used to design taxonomy (by volume)",
    300, 2000, 1000, 50,
    help="Use a larger sample (e.g., 1500–2000) for more hubs; smaller runs faster."
)

# Max hubs & regen/debug controls
max_hubs = st.sidebar.slider("Max hubs in taxonomy", 60, 250, 150, 10)
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
st.sidebar.caption("Tip: If you see too many 'Misc', lower strictness or increase sample/max hubs, then Regenerate taxonomy.")

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

# ──────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────
BANNED_LABELS = ["AI Education Hub", "A Level", "A-Level", "Misc", "General", "Education", "Academics"]

def taxonomy_prompt(sample_items: List[Dict[str, Any]], max_hubs_param: int) -> str:
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
- Avoid banned titles: {", ".join(BANNED_LABELS)}.
- Keep consistent level/mode/geo only if common across items.
- Favor hubs that align to user navigation; keep granularity consistent across hubs.
- IDs must be uppercase (A–Z, 0–9, underscores), unique and stable.
- Include short 'accepts' terms that characterize the hub; include 'avoid' for common confusions.
- Keep depth at most 2 (parent/child). Parent hubs optional.

Cluster summaries (sample):
{json.dumps(sample_items, ensure_ascii=False, indent=2)}

Return JSON only.
""".strip()

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
# Step 2 — Design taxonomy (hubs) from sample clusters
# ──────────────────────────────────────────────────────────────────────
st.subheader("Step 2 — Designing the taxonomy (hubs) from clusters")

# Select top-N clusters by total volume
sorted_clusters = sorted(cluster_total_vol.items(), key=lambda x: x[1], reverse=True)
sample = [{"cluster": c, "summary": cluster_summaries[c], "total_search_volume": int(v)}
          for c, v in sorted_clusters[:sample_top_n]]

# Token-safety: rough guard to avoid overly long prompts → truncated JSON
if len(sample) > 0:
    total_chars = sum(len(s["summary"]) for s in sample)
    if total_chars > 180_000:
        keep = max(400, int(180_000 / (total_chars / len(sample))))
        sample = sample[:keep]
        st.warning(f"Large sample trimmed to {keep} items to ensure valid JSON output.")

@st.cache_data(show_spinner=False)
def _compute_taxonomy(file_sig: str, sample_json: str, max_hubs_param: int, t: float) -> Dict[str, Any]:
    prompt = taxonomy_prompt(json.loads(sample_json), max_hubs_param=max_hubs_param)
    ia = call_openai_json(prompt, max_tokens=2000, temperature=t)
    return ia

file_sig = file_hash_bytes(csv_bytes) + f"|sample={len(sample)}|maxh={max_hubs}|t={temperature}"
sample_json = json.dumps(sample, ensure_ascii=False)

if regenerate_taxonomy:
    _compute_taxonomy.clear()

ia_raw = _compute_taxonomy(file_sig, sample_json, max_hubs, temperature)

# Basic validation & cleanup
hubs = ia_raw.get("hubs", [])
seen_ids, seen_titles = set(), set()
cleaned_hubs = []
for h in hubs:
    hid = str(h.get("id", "")).upper().strip().replace(" ", "_")
    hid = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in hid)
    title = str(h.get("title", "")).strip()
    if not hid or not title:
        continue
    if hid in seen_ids or title.lower() in seen_titles:
        continue
    if title in {"Misc","General","Education","AI Education Hub","A Level","A-Level"}:
        continue
    seen_ids.add(hid); seen_titles.add(title.lower())
    cleaned_hubs.append({
        "id": hid,
        "title": title,
        "parent_id": h.get("parent_id", None),
        "accepts": [a for a in h.get("accepts", [])[:20] if isinstance(a, str)],
        "avoid":   [a for a in h.get("avoid",   [])[:20] if isinstance(a, str)],
    })

ia = {"hubs": cleaned_hubs, "constraints": ia_raw.get("constraints", {"max_hubs": max_hubs, "max_depth": 2})}

if show_raw_taxonomy:
    st.code(json.dumps(ia_raw, ensure_ascii=False, indent=2)[:4000], language="json")

st.caption("Review/edit the generated taxonomy (optional).")
ia_text = st.text_area("Taxonomy JSON", json.dumps(ia, ensure_ascii=False, indent=2), height=320)
try:
    ia = json.loads(ia_text)
    hubs_list: List[Dict[str, Any]] = ia.get("hubs", [])
except Exception as e:
    st.error(f"Invalid JSON in taxonomy editor: {e}")
    st.stop()

if not hubs_list:
    st.error("No hubs were generated. Increase sample size or max hubs and regenerate.")
    st.stop()

# Build maps for hub metadata
hub_id_to_title  = {h["id"]: h["title"] for h in hubs_list}
hub_id_to_parent = {h["id"]: h.get("parent_id") for h in hubs_list}

# Hub summaries for similarity guard
hub_summaries = [f'{h["title"]} — ' + ", ".join([t for t in h.get("accepts", []) if isinstance(t, str)]) for h in hubs_list]
hub_ids_ordered    = [h["id"] for h in hubs_list]
hub_titles_ordered = [h["title"] for h in hubs_list]

st.caption("Embedding hub summaries for assignment sanity checks…")
hub_vecs = embed_texts_cached(tuple(hub_summaries))
hub_vecs = normalize(hub_vecs, axis=1)

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
        sims.append((best_id, float((cluster_vec @ hub_vecs[i].reshape(-1,1)).ravel()[0])))
    except ValueError:
        pass
    for aid in alt_ids[:3]:
        try:
            j = hub_ids_ordered.index(aid)
            sims.append((aid, float((cluster_vec @ hub_vecs[j].reshape(-1,1)).ravel()[0])))
        except ValueError:
            continue
    sims_sorted = sorted(sims, key=lambda x: (x[0] != best_id, -x[1]))  # prefer best_id then higher sim
    for hid, sim in sims_sorted:
        if sim >= sim_floor:
            return hid
    return best_id

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

    # Confidence handling
    chosen = best
    if confidence < conf_floor:
        chosen = fallback_parent(best)

    # Similarity guard
    try:
        cvec = embed_cluster_summary(summary)
        chosen = choose_with_similarity_guard(cvec, chosen, alt_ids)
    except Exception:
        pass

    out = {"best": best, "chosen": chosen, "confidence": confidence, "alts": alts}
    assign_cache[key] = out
    return out

# Parallel assignment with progress
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

# Optional: export assignment issues (helps debug "Misc"/failures)
issues = []
for cname, res in results_log:
    if isinstance(res, dict):
        if res.get("chosen") is None or hub_id_to_title.get(res.get("chosen","")) is None:
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





























