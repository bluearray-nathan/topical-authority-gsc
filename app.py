import re
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
import streamlit as st

# Optional (best) for unique counting at scale:
# pip install datasketch
try:
    from datasketch import HyperLogLogPlusPlus  # type: ignore
    HAS_HLL = True
except Exception:
    HAS_HLL = False

st.set_page_config(page_title="Embedding Based Internal Linking Tool", layout="wide")
st.title("🔗 Embedding Based Internal Linking Tool")
st.write(
    "Analyse contextual internal links from Screaming Frog exports and generate embedding-based internal linking recommendations."
)

show_tutorial = st.toggle("Show guided tutorial", value=False)
if show_tutorial:
    st.markdown("### Guided Tutorial")
    t0, t1, t2, t3, t4 = st.tabs(
        ["Overview", "1) Inlinks Analysis", "2) Embedding Recommendations", "3) Prioritisation", "Troubleshooting"]
    )
    with t0:
        st.markdown(
            """
This tool has two connected workflows:

1. **Inlinks analysis**
- Input: Screaming Frog **All Inlinks** export (`.csv`, `.csv.gz`, or `.zip`)
- What it does: filters out non-contextual links and summarises internal linking patterns
- Outputs:
  - top destination pages by contextual link frequency
  - top and secondary anchor text usage per destination
  - filter impact report (how many links each filter removed)
  - downloadable contextual link exports

2. **Embedding recommendations**
- Input: Screaming Frog embeddings export (`Extract embeddings from page content`) + contextual links baseline
- What it does: suggests semantically relevant internal links while excluding links that already exist contextually
- Outputs:
  - Source URL -> Recommended Target URL suggestions
  - similarity score for each recommendation
  - optional focus-destination priority metric in output
  - downloadable recommendations CSV

Recommended flow:
1. Run **Inlinks analysis** first and export filtered contextual links.
2. Run **Embedding recommendations** and use that filtered export as baseline to avoid duplicates.

Screaming Frog exports you need:
1. **All Inlinks CSV**
   - Crawl your site.
   - Open **Bulk Export -> Links -> All Inlinks**.
   - Save as CSV (or compress to `.csv.gz` / `.zip`).
2. **Embeddings CSV**
   - In Screaming Frog, enable and run AI embedding extraction for page content.
   - Ensure the crawl has the `Extract embeddings from page content` column populated.
   - Export the relevant URL report including:
     - `Address`
     - `Extract embeddings from page content`
   - Save as CSV.
"""
        )
    with t1:
        st.markdown(
            """
1. Select **Workflow -> Inlinks analysis**.
2. Upload Screaming Frog **All Inlinks** export (`.csv`, `.csv.gz`, `.zip`).
3. Keep **Contextual mode** enabled for recommended defaults.
4. Optional: adjust filters (parameters, pagination, anchor exclusions, external sources).
5. Click **Run analysis**.
6. Review outputs:
   - **Top destinations**
   - **Filter impact**
7. Download:
   - **full destination summary CSV**
   - **full filtered rows CSV** (enable this option before running).

How to get this file from Screaming Frog:
1. Run a crawl.
2. Go to **Bulk Export -> Links -> All Inlinks**.
3. Export to CSV.
"""
        )
    with t2:
        st.markdown(
            """
1. Select **Workflow -> Embedding recommendations**.
2. Upload embeddings CSV (with URL + embedding vector column).
3. Set **Contextual inlinks baseline**:
   - Use last filtered export from inlinks analysis, or
   - Upload filtered inlinks CSV.
4. Optional: upload **Focus URLs** (destination URLs) and a priority metric column.
5. Set controls:
   - **Recommendations per page**
   - **Minimum similarity**
   - **Max pages to process**
6. Click **Generate embedding recommendations**.
7. Download:
   - **link opportunities CSV**
   - **new recommendations CSV**
   - **skipped existing links CSV**.

How to get this file from Screaming Frog:
1. Run a crawl where AI embeddings are extracted for page content.
2. Export a CSV that includes:
   - URL column (for example `Address`)
   - embedding vector column (for example `Extract embeddings from page content`)
"""
        )
    with t3:
        st.markdown(
            """
- **Link Opportunities** combines:
  - new opportunities
  - already-existing contextual links (for audit)
- **Opportunity_Score**:
  - uses similarity and priority metric (if numeric)
  - falls back to similarity when no metric
  - set to `0` for already-existing links
- Suggested workflow:
1. Prioritise rows where `Already_Exists=False`.
2. Sort by `Opportunity_Score` descending.
3. Use focus metrics (e.g. conversions, clicks, revenue) to align with business priorities.
"""
        )
    with t4:
        st.markdown(
            """
- ZIP upload error with `__MACOSX`: the app auto-ignores metadata files. Re-upload after restart if needed.
- Missing columns: ensure inlinks file includes `Source` and `Destination`.
- Embeddings parse issues: verify vectors are comma-separated floats in one column.
- No recommendations generated:
  - lower **Minimum similarity**
  - increase **Max pages to process**
  - check focus destination URLs overlap your embeddings/contextual baseline URLs.
"""
        )
workflow_mode = st.radio(
    "Workflow",
    options=["Inlinks analysis", "Embedding recommendations"],
    horizontal=True,
)
ui_mode = st.radio(
    "View mode",
    options=["Simple", "Advanced"],
    horizontal=True,
    index=0,
)

if workflow_mode == "Inlinks analysis":
    with st.expander("How To Use: Inlinks analysis", expanded=True):
        st.markdown(
            """
1. Upload your Screaming Frog All Inlinks export (`.csv`, `.csv.gz`, or `.zip`).
2. Leave **Contextual mode** enabled for recommended defaults.
3. Optional: adjust exclusions (navigation, anchors, parameters, pagination, external sources).
4. Click **Run analysis**.
5. Review:
   - **Top destinations**: pages receiving contextual links.
   - **Filter impact**: how many links each filter removed.
6. Download outputs:
   - destination summary
   - filtered rows (if enabled)
"""
        )
else:
    with st.expander("How To Use: Embedding recommendations", expanded=True):
        st.markdown(
            """
1. Upload embeddings CSV from Screaming Frog AI extraction.
2. Choose contextual baseline links (previous filtered export or upload filtered inlinks CSV).
3. Optional: upload focus destination URLs + priority metric.
4. Set recommendation controls (per page, similarity, max pages).
5. Click **Generate embedding recommendations**.
6. Download the recommendation CSV.

Notes:
- Existing contextual links are automatically removed from recommendations.
- Candidate URLs are limited to the contextual baseline to avoid off-scope suggestions.
"""
        )

# Session state for persisted run results
if "analysis_output_df" not in st.session_state:
    st.session_state["analysis_output_df"] = None
if "total_links_found" not in st.session_state:
    st.session_state["total_links_found"] = 0
if "filtered_export_path" not in st.session_state:
    st.session_state["filtered_export_path"] = None
if "filter_removal_stats" not in st.session_state:
    st.session_state["filter_removal_stats"] = {}
if "total_rows_processed" not in st.session_state:
    st.session_state["total_rows_processed"] = 0
if "total_rows_kept" not in st.session_state:
    st.session_state["total_rows_kept"] = 0
if "embedding_recommendations_df" not in st.session_state:
    st.session_state["embedding_recommendations_df"] = None
if "embedding_recommendations_summary" not in st.session_state:
    st.session_state["embedding_recommendations_summary"] = ""
if "embedding_skipped_existing_df" not in st.session_state:
    st.session_state["embedding_skipped_existing_df"] = None
if "embedding_opportunities_df" not in st.session_state:
    st.session_state["embedding_opportunities_df"] = None
if "embedding_presentation_df" not in st.session_state:
    st.session_state["embedding_presentation_df"] = None
if "embedding_summary_metrics" not in st.session_state:
    st.session_state["embedding_summary_metrics"] = {}
if "inlinks_filter_config" not in st.session_state:
    st.session_state["inlinks_filter_config"] = None

# ---------------------------
# Helpers
# ---------------------------

def normalize_url_for_compare(u: str) -> str:
    """Light normalization for Source==Destination comparisons."""
    if u is None or pd.isna(u):
        return ""
    u = str(u).strip()
    if not u:
        return ""
    parts = urlsplit(u)
    scheme = (parts.scheme or "").lower()
    netloc = (parts.netloc or "").lower()
    path = parts.path or ""
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    # drop fragment
    return urlunsplit((scheme, netloc, path, parts.query, ""))

def extract_hostname(u: str) -> str:
    if u is None or pd.isna(u):
        return ""
    try:
        netloc = (urlsplit(str(u).strip()).netloc or "").lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""

def host_matches_domain(host: str, domain: str) -> bool:
    h = (host or "").lower().strip(".")
    d = (domain or "").lower().strip(".")
    if h.startswith("www."):
        h = h[4:]
    if d.startswith("www."):
        d = d[4:]
    return bool(h and d and (h == d or h.endswith(f".{d}")))

def infer_primary_domain(df: pd.DataFrame) -> str:
    """Infer site domain from most common destination host (fallback: source host)."""
    for col in ["Destination", "Source"]:
        if col not in df.columns:
            continue
        hosts = df[col].astype("string").fillna("").map(extract_hostname)
        hosts = hosts[hosts != ""]
        if not hosts.empty:
            counts = hosts.value_counts()
            return str(counts.index[0])
    return ""

def normalize_column_name(name: str) -> str:
    cleaned = str(name).replace("\ufeff", "").strip()
    cleaned = cleaned.strip('"').strip("'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping: dict[str, str] = {}
    for col in df.columns:
        normalized = normalize_column_name(col)
        mapping[col] = normalized if normalized else str(col)
    return df.rename(columns=mapping)


def compile_contains_patterns(lines: str) -> re.Pattern | None:
    """Each non-empty line is a substring match (escaped), OR'd together, case-insensitive."""
    pats = []
    for line in (lines or "").splitlines():
        line = line.strip()
        if not line:
            continue
        pats.append(re.escape(line))
    if not pats:
        return None
    return re.compile("|".join(pats), flags=re.IGNORECASE)

def passes_saved_inlinks_url_filters(url_value: str, cfg: dict | None) -> bool:
    if not cfg:
        return True
    u = str(url_value or "")
    if not u:
        return False
    if cfg.get("exclude_params", False) and "?" in u:
        return False
    paginated_patterns = cfg.get("paginated_patterns", "")
    if cfg.get("exclude_paginated_urls", False) and paginated_patterns:
        rx = compile_contains_patterns(str(paginated_patterns))
        if rx is not None and bool(re.search(rx, u)):
            return False
    destination_patterns = cfg.get("destination_patterns", "")
    if cfg.get("exclude_destination", False) and destination_patterns:
        rx = compile_contains_patterns(str(destination_patterns))
        if rx is not None and bool(re.search(rx, u)):
            return False
    allowed_domain = str(cfg.get("allowed_domain", "") or "").strip()
    if cfg.get("exclude_external_destinations", False) and allowed_domain:
        if not host_matches_domain(extract_hostname(u), allowed_domain):
            return False
    return True


def safe_str_series(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("")

def default_compression_from_filename(filename: str | None) -> str:
    if not filename:
        return "infer"
    lower_name = filename.lower()
    if lower_name.endswith(".gz"):
        return "gzip"
    return "infer"

def resolve_zip_archive_name(path: str) -> str:
    """Return the CSV member to read from a ZIP, ignoring macOS metadata files."""
    candidates: list[zipfile.ZipInfo] = []
    with zipfile.ZipFile(path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = member.filename
            base = PurePosixPath(name).name
            if name.startswith("__MACOSX/"):
                continue
            if base.startswith("._") or base.startswith("."):
                continue
            if name.lower().endswith(".csv"):
                candidates.append(member)

    if len(candidates) == 1:
        return candidates[0].filename

    if not candidates:
        raise ValueError("No CSV file found in ZIP. Include exactly one CSV file.")

    names = [c.filename for c in candidates]
    raise ValueError(f"Multiple CSV files found in ZIP. Keep only one CSV: {names}")

def compression_arg(path: str, hint: str):
    if hint == "none":
        return None
    return hint

def parse_embedding_vector(text: str) -> np.ndarray | None:
    if text is None or pd.isna(text):
        return None
    raw = str(text).strip().strip('"').strip("'")
    if not raw:
        return None
    try:
        vals = [float(x.strip()) for x in raw.split(",") if x.strip() != ""]
    except Exception:
        return None
    if not vals:
        return None
    return np.asarray(vals, dtype=np.float32)

def pick_first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        key = cand.lower()
        if key in lowered:
            return lowered[key]
    for c in columns:
        lc = c.lower()
        for cand in candidates:
            if cand.lower() in lc:
                return c
    return None

def load_existing_link_pairs(df: pd.DataFrame) -> set[tuple[str, str]]:
    working = canonicalize_columns(df.copy())
    src_col = pick_first_existing_column(working.columns.tolist(), ["Source", "From", "Source URL"])
    dst_col = pick_first_existing_column(working.columns.tolist(), ["Destination", "Target", "To", "URL"])
    if not src_col or not dst_col:
        return set()
    src_vals = working[src_col].astype("string").fillna("").map(normalize_url_for_compare)
    dst_vals = working[dst_col].astype("string").fillna("").map(normalize_url_for_compare)
    pairs: set[tuple[str, str]] = set()
    for s, d in zip(src_vals.tolist(), dst_vals.tolist()):
        if not s or not d:
            continue
        pairs.add((str(s), str(d)))
    return pairs

def load_allowed_urls_from_links(df: pd.DataFrame) -> set[str]:
    working = canonicalize_columns(df.copy())
    src_col = pick_first_existing_column(working.columns.tolist(), ["Source", "From", "Source URL"])
    dst_col = pick_first_existing_column(working.columns.tolist(), ["Destination", "Target", "To", "URL"])
    allowed: set[str] = set()
    if src_col:
        src_vals = working[src_col].astype("string").fillna("").map(normalize_url_for_compare)
        allowed.update({str(v) for v in src_vals.tolist() if str(v)})
    if dst_col:
        dst_vals = working[dst_col].astype("string").fillna("").map(normalize_url_for_compare)
        allowed.update({str(v) for v in dst_vals.tolist() if str(v)})
    return allowed

if workflow_mode == "Embedding recommendations":
    st.subheader("🧠 Embedding-Based Internal Linking Recommendations")
    st.caption("Upload Screaming Frog AI embeddings export, then generate semantically similar internal linking suggestions.")

    embeddings_file = st.file_uploader(
        "Upload embeddings CSV",
        type=["csv"],
        key="embeddings_upload",
    )
    st.markdown("### Contextual inlinks baseline (for deduplication)")
    dedupe_sources = ["Upload filtered inlinks CSV"]
    session_filtered_path = st.session_state.get("filtered_export_path")
    has_session_filtered = bool(session_filtered_path and Path(str(session_filtered_path)).exists())
    if has_session_filtered:
        dedupe_sources.insert(0, "Use last filtered export from inlinks analysis")
    dedupe_source = st.selectbox("Existing contextual links source", options=dedupe_sources)
    contextual_links_file = None
    if dedupe_source == "Upload filtered inlinks CSV":
        contextual_links_file = st.file_uploader(
            "Upload filtered inlinks CSV (must include Source and Destination)",
            type=["csv"],
            key="contextual_inlinks_upload",
        )
    preset = st.selectbox(
        "Recommendation preset",
        options=["Balanced", "Conservative", "Aggressive"],
        index=0,
        help="Balanced is recommended for most cases.",
    )
    preset_values = {
        "Conservative": {"k": 3, "min_similarity": 0.85, "max_pages": 700},
        "Balanced": {"k": 5, "min_similarity": 0.78, "max_pages": 1000},
        "Aggressive": {"k": 8, "min_similarity": 0.70, "max_pages": 1500},
    }
    preset_cfg = preset_values[preset]
    presentation_output = st.checkbox(
        "Executive output",
        value=True,
        help="Shows executive summary and implementation shortlist.",
    )

    focus_df_uploaded: pd.DataFrame | None = None
    focus_url_col = None
    focus_metric_col = None
    source_priority_df_uploaded: pd.DataFrame | None = None
    source_priority_url_col = None
    source_priority_metric_col = None
    with st.expander("Advanced recommendation options", expanded=(ui_mode == "Advanced")):
        st.markdown("### Focus URLs (optional)")
        st.caption("Upload a list of priority destination URLs and a metric. Recommendations will target these URLs.")
        focus_urls_file = st.file_uploader(
            "Upload focus URLs CSV",
            type=["csv"],
            key="focus_urls_upload",
        )
        if focus_urls_file is not None:
            focus_df_uploaded = canonicalize_columns(pd.read_csv(focus_urls_file, low_memory=False))
            focus_columns = focus_df_uploaded.columns.tolist()
            default_focus_url_col = pick_first_existing_column(
                focus_columns, ["Address", "URL", "Source", "Destination", "Page", "Landing Page"]
            ) or focus_columns[0]
            focus_url_col = st.selectbox(
                "Focus URL column",
                options=focus_columns,
                index=focus_columns.index(default_focus_url_col),
            )
            metric_options = ["(none)"] + [c for c in focus_columns if c != focus_url_col]
            default_metric_idx = 0
            if len(metric_options) > 1:
                preferred_metric = pick_first_existing_column(
                    metric_options, ["Priority", "Score", "Metric", "Clicks", "Conversions", "Revenue", "Traffic"]
                )
                if preferred_metric and preferred_metric in metric_options:
                    default_metric_idx = metric_options.index(preferred_metric)
            focus_metric_col = st.selectbox(
                "Priority metric column",
                options=metric_options,
                index=default_metric_idx,
                help="Any metric works (e.g. clicks, conversions, revenue, traffic).",
            )

        st.markdown("### Source URL priority (optional)")
        st.caption("Upload source URLs with a metric to prioritise which pages to edit first.")
        source_priority_file = st.file_uploader(
            "Upload source URL priority CSV",
            type=["csv"],
            key="source_priority_upload",
        )
        if source_priority_file is not None:
            source_priority_df_uploaded = canonicalize_columns(pd.read_csv(source_priority_file, low_memory=False))
            source_priority_columns = source_priority_df_uploaded.columns.tolist()
            default_source_url_col = pick_first_existing_column(
                source_priority_columns, ["Address", "URL", "Source", "Page", "Landing Page"]
            ) or source_priority_columns[0]
            source_priority_url_col = st.selectbox(
                "Source URL column",
                options=source_priority_columns,
                index=source_priority_columns.index(default_source_url_col),
            )
            source_metric_options = ["(none)"] + [c for c in source_priority_columns if c != source_priority_url_col]
            source_metric_default_idx = 0
            if len(source_metric_options) > 1:
                preferred_source_metric = pick_first_existing_column(
                    source_metric_options, ["Priority", "Score", "Metric", "Clicks", "Conversions", "Revenue", "Traffic"]
                )
                if preferred_source_metric and preferred_source_metric in source_metric_options:
                    source_metric_default_idx = source_metric_options.index(preferred_source_metric)
            source_priority_metric_col = st.selectbox(
                "Source priority metric column",
                options=source_metric_options,
                index=source_metric_default_idx,
                help="Any numeric metric can be used to prioritise source pages.",
            )

        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            recommendations_per_page = st.number_input(
                "Suggestions per source page",
                min_value=1,
                max_value=20,
                value=int(preset_cfg["k"]),
                step=1,
                help="Maximum number of suggested target URLs per source page.",
            )
        with col_e2:
            min_similarity = st.number_input(
                "Minimum similarity",
                min_value=0.0,
                max_value=1.0,
                value=float(preset_cfg["min_similarity"]),
                step=0.01,
                format="%.2f",
                help="Higher values are stricter (fewer but closer semantic matches).",
            )
        with col_e3:
            max_pages = st.number_input(
                "Page limit",
                min_value=10,
                max_value=5000,
                value=int(preset_cfg["max_pages"]),
                step=10,
                help="Limit used for performance on large exports.",
            )

    run_embeddings = st.button("🚀 Generate embedding recommendations", type="primary")

    if run_embeddings:
        if embeddings_file is None:
            st.error("Upload an embeddings CSV to continue.")
            st.stop()

        existing_pairs: set[tuple[str, str]] = set()
        allowed_urls: set[str] = set()
        if dedupe_source == "Use last filtered export from inlinks analysis" and has_session_filtered:
            existing_df = pd.read_csv(str(session_filtered_path), low_memory=False)
            existing_pairs = load_existing_link_pairs(existing_df)
            allowed_urls = load_allowed_urls_from_links(existing_df)
        elif dedupe_source == "Upload filtered inlinks CSV":
            if contextual_links_file is None:
                st.error("Upload a filtered inlinks CSV to deduplicate existing contextual links.")
                st.stop()
            existing_df = pd.read_csv(contextual_links_file, low_memory=False)
            existing_pairs = load_existing_link_pairs(existing_df)
            allowed_urls = load_allowed_urls_from_links(existing_df)

        if dedupe_source and not existing_pairs:
            st.warning("No existing Source→Destination pairs were detected in the contextual inlinks baseline.")
        if dedupe_source and not allowed_urls:
            st.error("No valid Source/Destination URLs found in the contextual inlinks baseline.")
            st.stop()
        saved_filter_cfg = st.session_state.get("inlinks_filter_config")

        focus_destination_norms: set[str] = set()
        focus_metric_by_url: dict[str, object] = {}
        focus_metric_numeric_by_url: dict[str, float] = {}
        active_destination_metric_col = None
        source_metric_by_url: dict[str, object] = {}
        source_metric_numeric_by_url: dict[str, float] = {}
        active_source_metric_col = None
        if focus_df_uploaded is not None:
            if not focus_url_col:
                st.error("Select a Focus URL column.")
                st.stop()
            focus_working = focus_df_uploaded.copy()
            url_series = focus_working[focus_url_col].astype("string").fillna("").map(normalize_url_for_compare)
            metric_series = (
                focus_working[focus_metric_col].astype("string").fillna("")
                if focus_metric_col and focus_metric_col != "(none)"
                else pd.Series([""] * len(focus_working), dtype="string")
            )
            metric_numeric = pd.to_numeric(metric_series, errors="coerce")
            active_destination_metric_col = focus_metric_col if focus_metric_col and focus_metric_col != "(none)" else None
            for u, m_raw, m_num in zip(url_series.tolist(), metric_series.tolist(), metric_numeric.tolist()):
                if not u:
                    continue
                u_norm = str(u)
                focus_destination_norms.add(u_norm)
                if active_destination_metric_col:
                    focus_metric_by_url[u_norm] = str(m_raw)
                    if pd.notna(m_num):
                        focus_metric_numeric_by_url[u_norm] = float(m_num)

        if source_priority_df_uploaded is not None:
            if not source_priority_url_col:
                st.error("Select a Source URL column for source priority.")
                st.stop()
            source_working = source_priority_df_uploaded.copy()
            source_url_series = source_working[source_priority_url_col].astype("string").fillna("").map(normalize_url_for_compare)
            source_metric_series = (
                source_working[source_priority_metric_col].astype("string").fillna("")
                if source_priority_metric_col and source_priority_metric_col != "(none)"
                else pd.Series([""] * len(source_working), dtype="string")
            )
            source_metric_numeric = pd.to_numeric(source_metric_series, errors="coerce")
            active_source_metric_col = (
                source_priority_metric_col if source_priority_metric_col and source_priority_metric_col != "(none)" else None
            )
            for u, m_raw, m_num in zip(source_url_series.tolist(), source_metric_series.tolist(), source_metric_numeric.tolist()):
                if not u:
                    continue
                u_norm = str(u)
                if active_source_metric_col:
                    source_metric_by_url[u_norm] = str(m_raw)
                    if pd.notna(m_num):
                        source_metric_numeric_by_url[u_norm] = float(m_num)

        emb_df = canonicalize_columns(pd.read_csv(embeddings_file, low_memory=False))
        url_col = pick_first_existing_column(emb_df.columns.tolist(), ["Address", "URL", "Destination", "Source"])
        embedding_col = pick_first_existing_column(
            emb_df.columns.tolist(),
            ["Extract embeddings from page content", "Embedding", "Embeddings", "Vector"],
        )

        if not url_col or not embedding_col:
            st.error("Could not detect required columns. Expected URL column (e.g. Address) and embedding column.")
            st.stop()

        working = emb_df[[url_col, embedding_col]].copy()
        working[url_col] = working[url_col].astype("string").fillna("").str.strip()
        working = working[working[url_col] != ""]
        working["_url_norm"] = working[url_col].map(normalize_url_for_compare)
        if saved_filter_cfg:
            working = working[working["_url_norm"].map(lambda u: passes_saved_inlinks_url_filters(str(u), saved_filter_cfg))]
        elif allowed_urls:
            working = working[working["_url_norm"].isin(allowed_urls)]
        working = working.drop_duplicates(subset=[url_col], keep="first")
        if len(working) > int(max_pages):
            working = working.head(int(max_pages))

        parsed_vectors: list[np.ndarray] = []
        kept_urls: list[str] = []
        kept_url_norms: list[str] = []
        for u, u_norm, v in zip(working[url_col].tolist(), working["_url_norm"].tolist(), working[embedding_col].tolist()):
            vec = parse_embedding_vector(v)
            if vec is None:
                continue
            kept_urls.append(str(u))
            kept_url_norms.append(str(u_norm))
            parsed_vectors.append(vec)

        if len(parsed_vectors) < 2:
            st.error("Not enough valid embedding rows to generate recommendations.")
            st.stop()

        dims = {vec.shape[0] for vec in parsed_vectors}
        if len(dims) != 1:
            st.error("Embedding vectors have inconsistent dimensions. Please export with a single embedding model.")
            st.stop()

        matrix = np.vstack(parsed_vectors)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms

        sim = matrix @ matrix.T
        sim = np.nan_to_num(sim, nan=-1.0, posinf=-1.0, neginf=-1.0)
        np.fill_diagonal(sim, -1.0)

        rec_rows: list[dict[str, object]] = []
        skipped_existing_rows: list[dict[str, object]] = []
        k = int(recommendations_per_page)
        min_sim = float(min_similarity)
        skipped_existing = 0
        source_indices = list(range(len(kept_urls)))
        if focus_destination_norms:
            matched_focus_destinations = {n for n in kept_url_norms if n in focus_destination_norms}
            if not matched_focus_destinations:
                st.error("None of the uploaded focus destination URLs matched the embeddings + contextual baseline URL set.")
                st.stop()

        for i in source_indices:
            src_url = kept_urls[i]
            row = sim[i]
            candidate_idx = np.argpartition(-row, kth=min(k * 5, len(row) - 1))[: min(k * 5, len(row))]
            ranked = sorted(
                ((int(j), float(row[j])) for j in candidate_idx if row[j] >= min_sim),
                key=lambda x: x[1],
                reverse=True,
            )
            selected_for_source = 0
            for j, score in ranked:
                src_norm = normalize_url_for_compare(src_url)
                dst_norm = normalize_url_for_compare(kept_urls[j])
                if focus_destination_norms and dst_norm not in focus_destination_norms:
                    continue
                if (src_norm, dst_norm) in existing_pairs:
                    skipped_existing += 1
                    skipped_row: dict[str, object] = {
                        "Source_URL": src_url,
                        "Recommended_Target_URL": kept_urls[j],
                        "Similarity": round(score, 4),
                        "Skip_Reason": "Already exists in contextual baseline",
                        "Already_Exists": True,
                        "Reason": "already_linked",
                    }
                    if active_destination_metric_col:
                        skipped_row[f"Destination Metric ({active_destination_metric_col})"] = focus_metric_by_url.get(dst_norm, "")
                        skipped_row["_destination_metric_num"] = focus_metric_numeric_by_url.get(dst_norm, np.nan)
                    if active_source_metric_col:
                        skipped_row[f"Source Metric ({active_source_metric_col})"] = source_metric_by_url.get(src_norm, "")
                        skipped_row["_source_metric_num"] = source_metric_numeric_by_url.get(src_norm, np.nan)
                    skipped_existing_rows.append(skipped_row)
                    continue
                rec_row: dict[str, object] = {
                    "Source_URL": src_url,
                    "Recommended_Target_URL": kept_urls[j],
                    "Similarity": round(score, 4),
                    "Already_Exists": False,
                    "Reason": "new_opportunity",
                }
                if active_destination_metric_col:
                    rec_row[f"Destination Metric ({active_destination_metric_col})"] = focus_metric_by_url.get(dst_norm, "")
                    rec_row["_destination_metric_num"] = focus_metric_numeric_by_url.get(dst_norm, np.nan)
                if active_source_metric_col:
                    rec_row[f"Source Metric ({active_source_metric_col})"] = source_metric_by_url.get(src_norm, "")
                    rec_row["_source_metric_num"] = source_metric_numeric_by_url.get(src_norm, np.nan)
                rec_rows.append(
                    rec_row
                )
                selected_for_source += 1
                if selected_for_source >= k:
                    break

        if not rec_rows:
            st.warning("No recommendations met the minimum similarity threshold.")
            st.stop()

        rec_df = pd.DataFrame(rec_rows)
        skipped_df = pd.DataFrame(skipped_existing_rows)
        if not rec_df.empty and ("_source_metric_num" in rec_df.columns or "_destination_metric_num" in rec_df.columns):
            rec_df = rec_df.sort_values(
                [c for c in ["_source_metric_num", "_destination_metric_num", "Similarity", "Source_URL"] if c in rec_df.columns],
                ascending=[False, False, False, True][: len([c for c in ["_source_metric_num", "_destination_metric_num", "Similarity", "Source_URL"] if c in rec_df.columns])],
                na_position="last",
            )
        else:
            rec_df = rec_df.sort_values(["Source_URL", "Similarity"], ascending=[True, False])
        if not skipped_df.empty and ("_source_metric_num" in skipped_df.columns or "_destination_metric_num" in skipped_df.columns):
            skipped_df = skipped_df.sort_values(
                [c for c in ["_source_metric_num", "_destination_metric_num", "Similarity", "Source_URL"] if c in skipped_df.columns],
                ascending=[False, False, False, True][: len([c for c in ["_source_metric_num", "_destination_metric_num", "Similarity", "Source_URL"] if c in skipped_df.columns])],
                na_position="last",
            )
        elif not skipped_df.empty:
            sort_cols = [c for c in ["Similarity", "Source_URL"] if c in skipped_df.columns]
            skipped_df = skipped_df.sort_values(sort_cols, ascending=[False, True] if len(sort_cols) == 2 else [False])

        opportunities_df = pd.concat([rec_df.copy(), skipped_df.copy()], ignore_index=True, sort=False)
        if not opportunities_df.empty:
            sim_component = pd.to_numeric(opportunities_df["Similarity"], errors="coerce").fillna(0.0)
            source_metric_norm = pd.Series(np.nan, index=opportunities_df.index, dtype="float64")
            destination_metric_norm = pd.Series(np.nan, index=opportunities_df.index, dtype="float64")
            if "_source_metric_num" in opportunities_df.columns:
                src_series = pd.to_numeric(opportunities_df["_source_metric_num"], errors="coerce")
                valid = src_series.notna()
                if valid.any():
                    min_v = float(src_series[valid].min())
                    max_v = float(src_series[valid].max())
                    source_metric_norm = (src_series - min_v) / (max_v - min_v) if max_v > min_v else pd.Series(1.0, index=src_series.index, dtype="float64")
            if "_destination_metric_num" in opportunities_df.columns:
                dst_series = pd.to_numeric(opportunities_df["_destination_metric_num"], errors="coerce")
                valid = dst_series.notna()
                if valid.any():
                    min_v = float(dst_series[valid].min())
                    max_v = float(dst_series[valid].max())
                    destination_metric_norm = (dst_series - min_v) / (max_v - min_v) if max_v > min_v else pd.Series(1.0, index=dst_series.index, dtype="float64")

            has_source = source_metric_norm.notna()
            has_destination = destination_metric_norm.notna()
            score = sim_component.copy()
            both_mask = has_source & has_destination
            src_only_mask = has_source & ~has_destination
            dst_only_mask = ~has_source & has_destination
            if both_mask.any():
                score.loc[both_mask] = (
                    0.6 * sim_component.loc[both_mask]
                    + 0.25 * source_metric_norm.loc[both_mask].fillna(0.0)
                    + 0.15 * destination_metric_norm.loc[both_mask].fillna(0.0)
                )
            if src_only_mask.any():
                score.loc[src_only_mask] = (
                    0.7 * sim_component.loc[src_only_mask]
                    + 0.3 * source_metric_norm.loc[src_only_mask].fillna(0.0)
                )
            if dst_only_mask.any():
                score.loc[dst_only_mask] = (
                    0.7 * sim_component.loc[dst_only_mask]
                    + 0.3 * destination_metric_norm.loc[dst_only_mask].fillna(0.0)
                )
            opportunities_df["Opportunity_Score"] = score
            opportunities_df.loc[opportunities_df["Already_Exists"] == True, "Opportunity_Score"] = 0.0
            opportunities_df["Opportunity_Score"] = opportunities_df["Opportunity_Score"].round(4)
            opportunities_df = opportunities_df.sort_values(
                ["Already_Exists", "Opportunity_Score", "Similarity"],
                ascending=[True, False, False],
            )

        for helper_col in ["_source_metric_num", "_destination_metric_num"]:
            if helper_col in rec_df.columns:
                rec_df = rec_df.drop(columns=[helper_col])
            if helper_col in skipped_df.columns:
                skipped_df = skipped_df.drop(columns=[helper_col])
            if helper_col in opportunities_df.columns:
                opportunities_df = opportunities_df.drop(columns=[helper_col])

        st.session_state["embedding_recommendations_df"] = rec_df
        st.session_state["embedding_skipped_existing_df"] = skipped_df
        st.session_state["embedding_opportunities_df"] = opportunities_df
        presentation_df = opportunities_df.copy()
        destination_metric_cols = [c for c in presentation_df.columns if c.startswith("Destination Metric (")]
        source_metric_cols = [c for c in presentation_df.columns if c.startswith("Source Metric (")]
        destination_metric_col = destination_metric_cols[0] if destination_metric_cols else None
        source_metric_col = source_metric_cols[0] if source_metric_cols else None
        if not presentation_df.empty:
            presentation_df["Relevance Score"] = (
                pd.to_numeric(presentation_df["Similarity"], errors="coerce").fillna(0.0) * 100.0
            ).round(1)
            presentation_df["Status"] = presentation_df["Reason"].map(
                {"new_opportunity": "New opportunity", "already_linked": "Already linked"}
            ).fillna("New opportunity")
            presentation_df["Why this recommendation"] = presentation_df.apply(
                lambda r: (
                    "High semantic relevance and strong implementation value."
                    if r["Status"] == "New opportunity" and float(r.get("Opportunity_Score", 0.0) or 0.0) >= 0.85
                    else (
                        "Strong semantic match with actionable relevance."
                        if r["Status"] == "New opportunity"
                        else "Already linked contextually; included for audit."
                    )
                ),
                axis=1,
            )

            rename_map = {
                "Source_URL": "Source URL",
                "Recommended_Target_URL": "Recommended Page",
                "Similarity": "Semantic Similarity",
                "Opportunity_Score": "Opportunity Score",
            }
            presentation_df = presentation_df.rename(columns=rename_map)

            ordered_cols = [
                "Source URL",
                "Recommended Page",
                "Status",
                "Opportunity Score",
                "Relevance Score",
                "Semantic Similarity",
            ]
            if source_metric_col and source_metric_col in presentation_df.columns:
                ordered_cols.append(source_metric_col)
            if destination_metric_col and destination_metric_col in presentation_df.columns:
                ordered_cols.append(destination_metric_col)
            ordered_cols.extend(["Why this recommendation"])
            ordered_cols = [c for c in ordered_cols if c in presentation_df.columns]
            presentation_df = presentation_df[ordered_cols + [c for c in presentation_df.columns if c not in ordered_cols]]

            summary = {
                "total_rows": int(len(presentation_df)),
                "new_opportunities": int((presentation_df["Status"] == "New opportunity").sum()),
                "already_linked": int((presentation_df["Status"] == "Already linked").sum()),
                "high_score_new": int(
                    ((presentation_df["Status"] == "New opportunity") & (pd.to_numeric(presentation_df["Opportunity Score"], errors="coerce") >= 0.85)).sum()
                ),
                "unique_source_pages": int(presentation_df["Source URL"].nunique() if "Source URL" in presentation_df.columns else 0),
                "unique_target_pages": int(presentation_df["Recommended Page"].nunique() if "Recommended Page" in presentation_df.columns else 0),
            }
        else:
            summary = {}

        st.session_state["embedding_presentation_df"] = presentation_df
        st.session_state["embedding_summary_metrics"] = summary
        st.session_state["embedding_recommendations_summary"] = (
            f"Generated {len(rec_df):,} recommendations from {len(source_indices):,} source pages "
            f"(candidate pool {len(kept_urls):,}, min similarity {min_sim:.2f}). "
            f"Skipped {skipped_existing:,} already-existing contextual links."
        )

    stored_rec_df = st.session_state.get("embedding_recommendations_df")
    if stored_rec_df is not None:
        st.success(st.session_state.get("embedding_recommendations_summary", "Embedding recommendations ready."))
        if presentation_output:
            summary_metrics = st.session_state.get("embedding_summary_metrics", {})
            presentation_df = st.session_state.get("embedding_presentation_df")
            if summary_metrics:
                st.subheader("Executive Summary")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("New opportunities", f"{summary_metrics.get('new_opportunities', 0):,}")
                k2.metric("High-score new", f"{summary_metrics.get('high_score_new', 0):,}")
                k3.metric("Source pages", f"{summary_metrics.get('unique_source_pages', 0):,}")
                k4.metric("Target pages", f"{summary_metrics.get('unique_target_pages', 0):,}")

            if presentation_df is not None and not presentation_df.empty:
                st.subheader("Opportunities Summary")
                st.dataframe(presentation_df, use_container_width=True)
                st.download_button(
                    "Download opportunities summary CSV",
                    data=presentation_df.to_csv(index=False).encode("utf-8"),
                    file_name="opportunities_summary.csv",
                    mime="text/csv",
                    on_click="ignore",
                )

        opportunities_df = st.session_state.get("embedding_opportunities_df")
        skipped_df = st.session_state.get("embedding_skipped_existing_df")

        # Keep embeddings results to a maximum of two tables for a simpler user experience.
        if not presentation_output and opportunities_df is not None and not opportunities_df.empty:
            st.subheader("Link Opportunities")
            st.caption("Includes new opportunities and already-existing contextual links for audit and prioritisation.")
            st.dataframe(opportunities_df, use_container_width=True)

        with st.expander("Downloads", expanded=True):
            if opportunities_df is not None and not opportunities_df.empty:
                st.download_button(
                    "Download link opportunities CSV",
                    data=opportunities_df.to_csv(index=False).encode("utf-8"),
                    file_name="embedding_link_opportunities.csv",
                    mime="text/csv",
                    on_click="ignore",
                )
            st.download_button(
                "Download embedding recommendations CSV",
                data=stored_rec_df.to_csv(index=False).encode("utf-8"),
                file_name="embedding_internal_link_recommendations.csv",
                mime="text/csv",
                on_click="ignore",
            )
            if skipped_df is not None and not skipped_df.empty:
                st.download_button(
                    "Download skipped existing links CSV",
                    data=skipped_df.to_csv(index=False).encode("utf-8"),
                    file_name="embedding_skipped_existing_links.csv",
                    mime="text/csv",
                    on_click="ignore",
                )

    st.stop()


# ---------------------------
# UI: Input source
# ---------------------------

st.subheader("📥 Input")
st.caption("Upload the full Screaming Frog All Inlinks export for contextual filtering and aggregation.")
uploaded_file = st.file_uploader(
    "Upload All Inlinks (.csv, .csv.gz, or .zip)",
    type=["csv", "gz", "zip"]
)

# Processing defaults (hidden from UI to keep experience simple)
chunksize = 200_000
sample_rows = 200_000
force_string_dtypes = True

# ---------------------------
# UI: Filters
# ---------------------------

st.subheader("🎛️ Filters")
contextual_mode = st.checkbox(
    "Contextual mode (recommended)",
    value=True,
    help="Keeps only likely in-content contextual links and removes common structural/navigation noise."
)

colA, colB, colC = st.columns([1, 1, 2])
with colA:
    remove_self = st.checkbox(
        "Remove self-referring links (Source == Destination)",
        value=True if contextual_mode else False,
        disabled=contextual_mode
    )
with colB:
    exclude_params = st.checkbox(
        "Exclude Destination URLs with query parameters ('?')",
        value=True if contextual_mode else False
    )
with colC:
    st.caption("These filters are applied during chunk processing (memory-friendly).")

apply_url_exclusions_to_source = st.checkbox(
    "Also apply parameter/pagination URL exclusions to Source URLs",
    value=False,
    help="When enabled, rows are removed if Source URL matches parameterized or paginated patterns."
)

with st.expander("Advanced URL/content filters", expanded=(ui_mode == "Advanced")):
    st.markdown("### 🧭 Exclude breadcrumb / structural navigation (Link Path patterns)")
    exclude_link_path = st.checkbox(
        "Enable Link Path exclusions",
        value=True if contextual_mode else False,
        disabled=contextual_mode
    )

    default_link_path_patterns = "\n".join([
        "breadcrumb",
        "/ol/li",
        "aria-label=\"breadcrumb\"",
        "aria-label='breadcrumb'",
    ])

    link_path_patterns = st.text_area(
        "Link Path patterns to EXCLUDE (one per line)",
        value=default_link_path_patterns,
        height=120,
        disabled=not exclude_link_path
    )

    st.markdown("### 🌐 Exclude navigational targets (Destination URL patterns)")
    exclude_destination = st.checkbox(
        "Enable Destination URL exclusions",
        value=True if contextual_mode else False,
        disabled=contextual_mode
    )

    default_dest_patterns = "\n".join([
        "/tag/",
        "/category/",
        "/collections/",
        "/search",
        "/login",
        "/account",
        "/cart",
        "/checkout",
    ])

    dest_patterns = st.text_area(
        "Destination patterns to EXCLUDE (one per line)",
        value=default_dest_patterns,
        height=140,
        disabled=not exclude_destination
    )

    st.markdown("### 📄 Exclude paginated destination URLs")
    exclude_paginated_urls = st.checkbox(
        "Enable paginated URL exclusions",
        value=True if contextual_mode else False,
    )

    default_paginated_patterns = "\n".join([
        "?page=",
        "&page=",
        "/page/",
        "/page-",
        "/paged/",
        "?paged=",
        "&paged=",
        "?pg=",
        "&pg=",
    ])

    paginated_patterns = st.text_area(
        "Paginated URL patterns to EXCLUDE (one per line)",
        value=default_paginated_patterns,
        height=120,
        disabled=not exclude_paginated_urls
    )

    st.markdown("### 🔤 Exclude generic anchor text")
    exclude_anchor_text = st.checkbox(
        "Enable Anchor text exclusions",
        value=True
    )

    default_anchor_patterns = "\n".join([
        "read more",
        "learn more",
        "click here",
        "find out more",
        "view more",
        "more",
        "here",
    ])

    anchor_text_patterns = st.text_area(
        "Anchor text patterns to EXCLUDE (one per line)",
        value=default_anchor_patterns,
        height=140,
        disabled=not exclude_anchor_text
    )

FILTER_COLUMNS = ["Type", "Follow", "Status Code", "Status", "Link Position", "Link Origin", "Target", "Rel", "Path Type"]

def contextual_default_display_values(col: str, display_vals: list[str], contextual_enabled: bool) -> list[str]:
    if not contextual_enabled:
        return display_vals

    preferred_by_col = {
        "Type": ["Hyperlink"],
        "Link Position": ["Content"],
        "Status Code": ["200"],
        "Status": ["OK"],
        "Follow": ["true"],
    }
    preferred = preferred_by_col.get(col, [])
    selected = [v for v in preferred if v in display_vals]
    return selected if selected else display_vals

with st.expander("Advanced output options", expanded=(ui_mode == "Advanced")):
    # Deduped ranking option (no global dedupe set needed)
    use_deduped_ranking = st.checkbox(
        "Rank by 'Unique Source Pages' (deduped Source→Destination) instead of raw occurrences",
        value=False,
        help=(
            "True global deduplication of Source→Destination pairs across an entire huge file can be memory-heavy. "
            "Ranking by Unique Source Pages gives you the 'deduped' view you usually want."
        )
    )
    export_filtered_rows = st.checkbox(
        "Enable full filtered CSV download",
        value=False,
        help="Creates a downloadable CSV containing all rows that remain after filters are applied."
    )

# ---------------------------
# Load uploaded file to temp path
# ---------------------------

def materialize_input_to_tempfile() -> tuple[str | None, str]:
    """
    Save uploaded file to a temp file and return (path, default compression hint).
    We do this so pandas can chunk-read reliably.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp_path = tmp.name
    tmp.close()

    if uploaded_file is None:
        return None, "infer"

    # write bytes to disk
    data = uploaded_file.getbuffer()
    Path(tmp_path).write_bytes(data)
    if zipfile.is_zipfile(tmp_path):
        csv_member = resolve_zip_archive_name(tmp_path)
        extracted = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        extracted_path = extracted.name
        extracted.close()
        with zipfile.ZipFile(tmp_path) as zf, zf.open(csv_member) as src, Path(extracted_path).open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return extracted_path, "none"

    return tmp_path, default_compression_from_filename(uploaded_file.name if uploaded_file else None)


tmp_path, default_compression_hint = materialize_input_to_tempfile()
if tmp_path is None:
    st.info("Provide an upload to continue.")
    st.stop()

# Try to infer compression from filename contents (uploaded gz is still saved as .csv)
# pandas can infer gz from file content? not reliably. We'll allow user hint:
compression_hint = st.selectbox(
    "File compression",
    options=["infer", "gzip", "none"],
    index=["infer", "gzip", "none"].index(default_compression_hint),
    help="For ZIP uploads, the app auto-extracts the CSV before processing."
)

# ---------------------------
# Build sample for column filter pickers
# ---------------------------

@st.cache_data(show_spinner=False)
def read_sample(path: str, nrows: int, compression_hint: str, force_str: bool) -> pd.DataFrame:
    read_kwargs = dict(
        nrows=nrows,
        low_memory=False,
        compression=compression_arg(path, compression_hint),
    )
    if force_str:
        read_kwargs["dtype"] = "string"

    # encoding fallback
    try:
        return canonicalize_columns(pd.read_csv(path, **read_kwargs))
    except UnicodeDecodeError:
        read_kwargs["encoding"] = "cp1252"
        return canonicalize_columns(pd.read_csv(path, **read_kwargs))

try:
    sample_df = read_sample(tmp_path, int(sample_rows), compression_hint, force_string_dtypes)
except ValueError as e:
    st.error(str(e))
    st.stop()

# Validate core columns
for core in ["Source", "Destination"]:
    if core not in sample_df.columns:
        st.error(f"Missing expected column in file sample: {core}")
        st.stop()

source_domain_default = infer_primary_domain(sample_df)

with st.expander("Advanced source/column filters", expanded=(ui_mode == "Advanced")):
    st.markdown("### 🌍 Source domain filter")
    exclude_external_sources = st.checkbox(
        "Exclude external Source URLs",
        value=True if contextual_mode else False,
    )
    exclude_external_destinations = st.checkbox(
        "Exclude external Destination URLs",
        value=True if contextual_mode else False,
    )
    allowed_source_domain = st.text_input(
        "Allowed domain (subdomains included)",
        value=source_domain_default,
        disabled=(not exclude_external_sources and not exclude_external_destinations and not contextual_mode),
        help="Example: rac.co.uk. Keeps URLs on this domain and its subdomains."
    )

    st.markdown("### 🧾 Column include filters (sample-based)")
    st.caption(
        "Because we process in chunks, we populate these dropdowns from a **sample** of the file "
        "(first N rows). This keeps the app fast and Cloud-safe."
    )

    # Build sample-based multiselects
    selected_values_by_col: dict[str, set[str]] = {}
    for col in FILTER_COLUMNS:
        if col in sample_df.columns:
            vals = sorted(sample_df[col].astype("string").fillna("").unique().tolist(), key=lambda x: str(x).lower())
            display_vals = ["(blank)" if str(v).strip() == "" else str(v) for v in vals]
            default_display_vals = contextual_default_display_values(col, display_vals, contextual_mode)
            selected_display = st.multiselect(
                f"Include values for **{col}** (sample-based)",
                options=display_vals,
                default=default_display_vals
            )
            # map display -> real string
            selected_real = set("" if d == "(blank)" else d for d in selected_display)
            selected_values_by_col[col] = selected_real

if "exclude_external_sources" not in locals():
    exclude_external_sources = True if contextual_mode else False
if "exclude_external_destinations" not in locals():
    exclude_external_destinations = True if contextual_mode else False
if "allowed_source_domain" not in locals():
    allowed_source_domain = source_domain_default
if "selected_values_by_col" not in locals():
    selected_values_by_col = {}

# Compile pattern regexes
rx_link_path = compile_contains_patterns(link_path_patterns) if exclude_link_path else None
rx_dest = compile_contains_patterns(dest_patterns) if exclude_destination else None
rx_paginated = compile_contains_patterns(paginated_patterns) if exclude_paginated_urls else None
rx_anchor = compile_contains_patterns(anchor_text_patterns) if exclude_anchor_text else None
rx_link_path_default = compile_contains_patterns(default_link_path_patterns)
rx_dest_default = compile_contains_patterns(default_dest_patterns)

# ---------------------------
# Chunk processing + aggregation
# ---------------------------

# Always use best available unique counting strategy.
unique_strategy = "hll" if HAS_HLL else "exact"

# Aggregates
# total occurrences per destination (raw rows after filters)
total_inlinks: dict[str, int] = {}
# anchor text counts per destination
anchor_counts_by_dest: dict[str, dict[str, int]] = {}

# unique source pages per destination
# - exact: store set of sources per dest (can be huge)
# - hll: store HyperLogLog++ per dest (memory light)
unique_sources_exact: dict[str, set[str]] | None = {} if unique_strategy == "exact" else None
unique_sources_hll: dict[str, "HyperLogLogPlusPlus"] | None = {} if unique_strategy == "hll" else None

def ensure_hll_for_dest(dest: str) -> "HyperLogLogPlusPlus":
    assert unique_sources_hll is not None
    h = unique_sources_hll.get(dest)
    if h is None:
        # p=12 is a good balance; you can raise to 14 for more accuracy (more memory).
        h = HyperLogLogPlusPlus(p=12)
        unique_sources_hll[dest] = h
    return h

def top_two_anchors(anchor_counts: dict[str, int], total_links: int) -> tuple[str, float, str, float]:
    if not anchor_counts or total_links <= 0:
        return "", 0.0, "", 0.0
    ranked = sorted(anchor_counts.items(), key=lambda x: (-x[1], x[0].lower()))
    top_text, top_count = ranked[0]
    top_pct = (top_count / total_links) * 100.0
    if len(ranked) > 1:
        second_text, second_count = ranked[1]
        second_pct = (second_count / total_links) * 100.0
    else:
        second_text, second_pct = "", 0.0
    return top_text, top_pct, second_text, second_pct

def build_destination_export_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "Destination Page",
                "Contextual Internal Links",
                "Unique Linking Pages",
                "Share of Contextual Links (%)",
                "Primary Anchor Text",
                "Primary Anchor Usage (%)",
                "Secondary Anchor Text",
                "Secondary Anchor Usage (%)",
            ]
        )

    out = df.copy()
    total_links = float(pd.to_numeric(out.get("Total_Inlink_Occurrences", 0), errors="coerce").fillna(0).sum())
    links_series = pd.to_numeric(out["Total_Inlink_Occurrences"], errors="coerce").fillna(0.0)

    share_pct = (links_series / total_links * 100.0) if total_links > 0 else pd.Series(0.0, index=out.index)
    exported = pd.DataFrame(
        {
            "Destination Page": out["Destination"],
            "Contextual Internal Links": links_series.astype(int),
            "Unique Linking Pages": pd.to_numeric(out["Unique_Source_Pages"], errors="coerce").fillna(0).astype(int),
            "Share of Contextual Links (%)": share_pct.round(2),
            "Primary Anchor Text": out["Top_Anchor_Text"].fillna("").astype(str),
            "Primary Anchor Usage (%)": pd.to_numeric(out["Top_Anchor_Usage_Pct"], errors="coerce").fillna(0.0).round(2),
            "Secondary Anchor Text": out["Secondary_Anchor_Text"].fillna("").astype(str),
            "Secondary Anchor Usage (%)": pd.to_numeric(out["Secondary_Anchor_Usage_Pct"], errors="coerce").fillna(0.0).round(2),
        }
    )
    return exported

def apply_filters_to_chunk(chunk: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    chunk = canonicalize_columns(chunk)
    removal_stats: dict[str, int] = {}

    def add_removed(label: str, before_count: int, after_count: int) -> None:
        removed = before_count - after_count
        if removed > 0:
            removal_stats[label] = removal_stats.get(label, 0) + removed

    # Ensure strings for operations
    # (If dtype already string, this is cheap)
    for c in ["Source", "Destination"]:
        chunk[c] = chunk[c].astype("string").fillna("")

    if contextual_mode:
        if "Type" in chunk.columns:
            before = len(chunk)
            type_series = chunk["Type"].astype("string").fillna("").str.strip().str.lower()
            chunk = chunk[type_series == "hyperlink"]
            add_removed("Contextual mode: keep Type = Hyperlink", before, len(chunk))
        if "Link Position" in chunk.columns:
            before = len(chunk)
            position_series = chunk["Link Position"].astype("string").fillna("").str.strip().str.lower()
            chunk = chunk[position_series == "content"]
            add_removed("Contextual mode: keep Link Position = Content", before, len(chunk))

    effective_remove_self = contextual_mode or remove_self
    effective_exclude_params = exclude_params
    effective_rx_link_path = rx_link_path_default if contextual_mode else rx_link_path
    effective_rx_dest = rx_dest_default if contextual_mode else rx_dest
    effective_rx_paginated = rx_paginated
    effective_rx_anchor = rx_anchor
    effective_source_domain_filter = (contextual_mode or exclude_external_sources) and bool(allowed_source_domain.strip())
    effective_destination_domain_filter = (contextual_mode or exclude_external_destinations) and bool(allowed_source_domain.strip())

    # Remove self-referring
    if effective_remove_self:
        before = len(chunk)
        src_norm = chunk["Source"].map(normalize_url_for_compare)
        dst_norm = chunk["Destination"].map(normalize_url_for_compare)
        chunk = chunk[src_norm != dst_norm]
        add_removed("Remove self-referring links", before, len(chunk))

    # Exclude params
    if effective_exclude_params:
        before = len(chunk)
        chunk = chunk[~chunk["Destination"].str.contains(r"\?", regex=True, na=False)]
        add_removed("Exclude destination query parameters", before, len(chunk))
        if apply_url_exclusions_to_source:
            before = len(chunk)
            chunk = chunk[~chunk["Source"].str.contains(r"\?", regex=True, na=False)]
            add_removed("Exclude source query parameters", before, len(chunk))

    # Link Path exclusion
    if effective_rx_link_path is not None and "Link Path" in chunk.columns:
        before = len(chunk)
        lp = chunk["Link Path"].astype("string").fillna("")
        chunk = chunk[~lp.str.contains(effective_rx_link_path, na=False)]
        add_removed("Exclude Link Path patterns", before, len(chunk))

    # Destination exclusion
    if effective_rx_dest is not None:
        before = len(chunk)
        chunk = chunk[~chunk["Destination"].str.contains(effective_rx_dest, na=False)]
        add_removed("Exclude destination URL patterns", before, len(chunk))

    # Paginated destination exclusion
    if effective_rx_paginated is not None:
        before = len(chunk)
        chunk = chunk[~chunk["Destination"].str.contains(effective_rx_paginated, na=False)]
        add_removed("Exclude paginated destination URLs", before, len(chunk))
        if apply_url_exclusions_to_source:
            before = len(chunk)
            chunk = chunk[~chunk["Source"].str.contains(effective_rx_paginated, na=False)]
            add_removed("Exclude paginated source URLs", before, len(chunk))

    # External source exclusion
    if effective_source_domain_filter and "Source" in chunk.columns:
        before = len(chunk)
        source_hosts = chunk["Source"].astype("string").fillna("").map(extract_hostname)
        allowed_domain = allowed_source_domain.strip()
        chunk = chunk[source_hosts.map(lambda h: host_matches_domain(str(h), allowed_domain))]
        add_removed("Exclude external Source URLs", before, len(chunk))

    # External destination exclusion
    if effective_destination_domain_filter and "Destination" in chunk.columns:
        before = len(chunk)
        destination_hosts = chunk["Destination"].astype("string").fillna("").map(extract_hostname)
        allowed_domain = allowed_source_domain.strip()
        chunk = chunk[destination_hosts.map(lambda h: host_matches_domain(str(h), allowed_domain))]
        add_removed("Exclude external Destination URLs", before, len(chunk))

    # Anchor text exclusion
    if effective_rx_anchor is not None and "Anchor" in chunk.columns:
        before = len(chunk)
        anchor_series = chunk["Anchor"].astype("string").fillna("")
        chunk = chunk[~anchor_series.str.contains(effective_rx_anchor, na=False)]
        add_removed("Exclude anchor text patterns", before, len(chunk))

    # Column include filters (from sample selections)
    for col, allowed in selected_values_by_col.items():
        if col in chunk.columns:
            before = len(chunk)
            s = chunk[col].astype("string").fillna("")
            chunk = chunk[s.isin(allowed)]
            add_removed(f"Include filter: {col}", before, len(chunk))

    return chunk, removal_stats

def process_file_in_chunks(path: str, write_filtered_csv: bool = False) -> tuple[str | None, dict[str, int], int, int]:
    read_kwargs = dict(
        chunksize=int(chunksize),
        low_memory=False,
        compression=compression_arg(path, compression_hint),
    )
    if force_string_dtypes:
        read_kwargs["dtype"] = "string"

    # encoding fallback
    try:
        iterator = pd.read_csv(path, **read_kwargs)
    except UnicodeDecodeError:
        read_kwargs["encoding"] = "cp1252"
        iterator = pd.read_csv(path, **read_kwargs)

    progress = st.progress(0, text="Processing chunks...")
    total_rows_seen = 0
    total_rows_kept = 0
    chunks_seen = 0
    filtered_export_path: str | None = None
    wrote_filtered_header = False
    cumulative_filter_stats: dict[str, int] = {}

    if write_filtered_csv:
        filtered_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        filtered_export_path = filtered_tmp.name
        filtered_tmp.close()

    for chunk in iterator:
        chunks_seen += 1
        total_rows_seen += len(chunk)
        chunk = canonicalize_columns(chunk)

        # Require columns exist in chunk
        if "Source" not in chunk.columns or "Destination" not in chunk.columns:
            continue

        filtered, chunk_filter_stats = apply_filters_to_chunk(chunk)
        for label, removed in chunk_filter_stats.items():
            cumulative_filter_stats[label] = cumulative_filter_stats.get(label, 0) + removed
        if filtered.empty:
            # Update progress text only
            if chunks_seen % 5 == 0:
                progress.progress(min(0.99, chunks_seen / (chunks_seen + 50)), text=f"Processed {total_rows_seen:,} rows...")
            continue
        total_rows_kept += len(filtered)

        if write_filtered_csv and filtered_export_path:
            filtered.to_csv(
                filtered_export_path,
                mode="a",
                index=False,
                header=not wrote_filtered_header
            )
            wrote_filtered_header = True

        # Aggregate totals
        dest_series = filtered["Destination"].astype("string").fillna("")
        src_series = filtered["Source"].astype("string").fillna("")
        anchor_series = (
            filtered["Anchor"].astype("string").fillna("").str.strip()
            if "Anchor" in filtered.columns
            else pd.Series([""] * len(filtered), dtype="string")
        )

        # Total inlink occurrences (raw rows after filters)
        counts = dest_series.value_counts(dropna=False)
        for dest, cnt in counts.items():
            dest = str(dest)
            total_inlinks[dest] = total_inlinks.get(dest, 0) + int(cnt)

        for dest, anchor in zip(dest_series.tolist(), anchor_series.tolist()):
            d = str(dest)
            a = str(anchor).strip()
            if not d or not a:
                continue
            per_dest = anchor_counts_by_dest.setdefault(d, {})
            per_dest[a] = per_dest.get(a, 0) + 1

        # Unique sources per destination
        if unique_strategy == "exact":
            assert unique_sources_exact is not None
            # WARNING: can grow large
            # We update sets per destination
            for dest, src in zip(dest_series.tolist(), src_series.tolist()):
                d = str(dest)
                s = str(src)
                if not d:
                    continue
                unique_sources_exact.setdefault(d, set()).add(s)

        else:
            # HyperLogLog++ (recommended for scale)
            assert unique_sources_hll is not None
            # Add hashed bytes (HLL expects bytes-like items)
            for dest, src in zip(dest_series.tolist(), src_series.tolist()):
                d = str(dest)
                s = str(src)
                if not d:
                    continue
                h = ensure_hll_for_dest(d)
                h.update(s.encode("utf-8", errors="ignore"))

        # Progress (rough; avoids needing total rows)
        if chunks_seen % 5 == 0:
            progress.progress(min(0.99, chunks_seen / (chunks_seen + 50)), text=f"Processed {total_rows_seen:,} rows...")

    progress.progress(1.0, text=f"Done. Processed {total_rows_seen:,} rows in {chunks_seen} chunks.")
    return filtered_export_path, cumulative_filter_stats, total_rows_seen, total_rows_kept

# Run processing
run = st.button("🚀 Run analysis", type="primary")
st.caption("The run may take time for large files. Progress is shown during chunk processing.")

if run:
    # Reset aggregates for a fresh run
    total_inlinks.clear()
    anchor_counts_by_dest.clear()
    if unique_sources_exact is not None:
        unique_sources_exact.clear()
    if unique_sources_hll is not None:
        unique_sources_hll.clear()
    existing_export_path = st.session_state.get("filtered_export_path")
    if existing_export_path:
        Path(existing_export_path).unlink(missing_ok=True)
        st.session_state["filtered_export_path"] = None
    st.session_state["filter_removal_stats"] = {}
    st.session_state["total_rows_processed"] = 0
    st.session_state["total_rows_kept"] = 0

    with st.spinner("Processing file in chunks..."):
        filtered_export_path, filter_removal_stats, total_rows_processed, total_rows_kept = process_file_in_chunks(
            tmp_path, write_filtered_csv=export_filtered_rows
        )

    # Build output table
    destinations = list(total_inlinks.keys())

    if unique_strategy == "exact":
        assert unique_sources_exact is not None
        unique_counts = {d: len(unique_sources_exact.get(d, set())) for d in destinations}
    else:
        assert unique_sources_hll is not None
        unique_counts = {d: int(unique_sources_hll[d].count()) if d in unique_sources_hll else 0 for d in destinations}

    out = pd.DataFrame({
        "Destination": destinations,
        "Total_Inlink_Occurrences": [total_inlinks.get(d, 0) for d in destinations],
        "Unique_Source_Pages": [unique_counts.get(d, 0) for d in destinations],
    })

    anchor_stats = [top_two_anchors(anchor_counts_by_dest.get(d, {}), total_inlinks.get(d, 0)) for d in destinations]
    out["Top_Anchor_Text"] = [row[0] for row in anchor_stats]
    out["Top_Anchor_Usage_Pct"] = [round(row[1], 2) for row in anchor_stats]
    out["Secondary_Anchor_Text"] = [row[2] for row in anchor_stats]
    out["Secondary_Anchor_Usage_Pct"] = [round(row[3], 2) for row in anchor_stats]

    # Ranking choice
    if use_deduped_ranking:
        out = out.sort_values(["Unique_Source_Pages", "Total_Inlink_Occurrences"], ascending=False)
        st.info("Ranking by **Unique_Source_Pages** (deduped view).")
    else:
        out = out.sort_values(["Total_Inlink_Occurrences", "Unique_Source_Pages"], ascending=False)
        st.info("Ranking by **Total_Inlink_Occurrences** (raw rows).")

    st.session_state["analysis_output_df"] = out
    st.session_state["total_links_found"] = int(out["Total_Inlink_Occurrences"].sum()) if not out.empty else 0
    st.session_state["filtered_export_path"] = filtered_export_path if export_filtered_rows else None
    st.session_state["filter_removal_stats"] = filter_removal_stats
    st.session_state["total_rows_processed"] = total_rows_processed
    st.session_state["total_rows_kept"] = total_rows_kept
    st.session_state["inlinks_filter_config"] = {
        "exclude_params": bool(exclude_params),
        "exclude_paginated_urls": bool(exclude_paginated_urls),
        "paginated_patterns": str(paginated_patterns),
        "exclude_destination": bool(exclude_destination),
        "destination_patterns": str(dest_patterns),
        "exclude_external_destinations": bool(exclude_external_destinations),
        "allowed_domain": str(allowed_source_domain or ""),
    }

if st.session_state["analysis_output_df"] is not None:
    out = st.session_state["analysis_output_df"]
    out_export = build_destination_export_df(out)
    if use_deduped_ranking:
        st.info("Ranking by **Unique_Source_Pages** (deduped view).")
    else:
        st.info("Ranking by **Total_Inlink_Occurrences** (raw rows).")

    top_n = st.number_input("Rows to show", min_value=10, max_value=500, value=50, step=10)
    out_top = out.head(int(top_n))
    out_top_export = out_export.head(int(top_n))
    out_top_display = out_top.copy()
    for col in ["Top_Anchor_Usage_Pct", "Secondary_Anchor_Usage_Pct"]:
        if col in out_top_display.columns:
            out_top_display[col] = out_top_display[col].map(lambda v: f"{float(v):.2f}%")

    st.subheader("🏆 Top destinations")
    st.caption("These are destination pages receiving the most contextual internal links after filters.")
    st.dataframe(out_top_display, use_container_width=True)

    filter_stats = st.session_state.get("filter_removal_stats", {})
    rows_processed = int(st.session_state.get("total_rows_processed", 0))
    rows_kept = int(st.session_state.get("total_rows_kept", 0))
    rows_removed = max(0, rows_processed - rows_kept)
    if ui_mode == "Simple":
        with st.expander("Filter impact details", expanded=False):
            st.caption(
                f"Rows processed: {rows_processed:,} | Rows kept: {rows_kept:,} | Rows removed: {rows_removed:,}"
            )
            if filter_stats:
                impact_df = pd.DataFrame(
                    [{"Filter": name, "Links Removed": count} for name, count in filter_stats.items()]
                ).sort_values("Links Removed", ascending=False)
                st.dataframe(impact_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No active filters removed any links in this run.")
    else:
        st.subheader("🧹 Filter impact")
        st.caption(
            f"Rows processed: {rows_processed:,} | Rows kept: {rows_kept:,} | Rows removed: {rows_removed:,}"
        )
        if filter_stats:
            impact_df = pd.DataFrame(
                [{"Filter": name, "Links Removed": count} for name, count in filter_stats.items()]
            ).sort_values("Links Removed", ascending=False)
            st.dataframe(impact_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No active filters removed any links in this run.")

    # Downloads
    st.subheader("⬇️ Downloads")
    st.caption("Use these exports for reporting or as input to embedding recommendation workflow.")
    c1, c2 = st.columns(2)

    with c1:
        st.download_button(
            "Download top destinations CSV",
            data=out_top_export.to_csv(index=False).encode("utf-8"),
            file_name="top_destinations_summary.csv",
            mime="text/csv",
            on_click="ignore",
        )

    with c2:
        st.download_button(
            "Download full destination summary CSV",
            data=out_export.to_csv(index=False).encode("utf-8"),
            file_name="destinations_summary.csv",
            mime="text/csv",
            on_click="ignore",
        )

    filtered_export_path = st.session_state.get("filtered_export_path")
    if filtered_export_path and Path(filtered_export_path).exists():
        with Path(filtered_export_path).open("rb") as filtered_file:
            st.download_button(
                "Download full filtered rows CSV",
                data=filtered_file,
                file_name="filtered_inlinks.csv",
                mime="text/csv",
                on_click="ignore",
            )

with st.expander("Performance Metrics", expanded=False):
    st.metric("Total links found (session)", f"{st.session_state['total_links_found']:,}")
    if st.button("Clear Cache", use_container_width=True):
        st.cache_data.clear()
        existing_export_path = st.session_state.get("filtered_export_path")
        if existing_export_path:
            Path(existing_export_path).unlink(missing_ok=True)
        st.session_state["analysis_output_df"] = None
        st.session_state["total_links_found"] = 0
        st.session_state["filtered_export_path"] = None
        st.session_state["filter_removal_stats"] = {}
        st.session_state["total_rows_processed"] = 0
        st.session_state["total_rows_kept"] = 0
        st.session_state["inlinks_filter_config"] = None
        st.rerun()

# Cleanup note
with st.expander("Notes / How to use for very large files"):
    st.markdown(
        """
- **Best practice:** upload **`.csv.gz`** (compress your CSV). All Inlinks files compress very well.
- You can also upload a **`.zip`** (single CSV inside).
- **Chunk processing** avoids loading the whole file into memory.
- **Unique Source Pages**:
  - If `datasketch` is installed, the app uses **HyperLogLog++** for large-file-safe approximate unique counts.
  - Exact unique counts can be memory heavy on big crawls.
"""
    )

# Best-effort temp file cleanup (optional)
# Streamlit reruns frequently; deleting immediately can break subsequent runs.
# You can leave temp files or implement periodic cleanup if deploying elsewhere.
