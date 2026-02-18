import re
import io
import os
import tempfile
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import streamlit as st

# Optional (best) for unique counting at scale:
# pip install datasketch
try:
    from datasketch import HyperLogLogPlusPlus  # type: ignore
    HAS_HLL = True
except Exception:
    HAS_HLL = False

# Optional for Drive download
try:
    import requests  # type: ignore
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False


st.set_page_config(page_title="All Inlinks Analyzer (Chunked)", layout="wide")
st.title("🔗 Screaming Frog All Inlinks Analyzer (Big-file friendly)")
st.write(
    "Handles large All Inlinks exports by processing in **chunks**. "
    "Upload a `.csv`/`.csv.gz` or paste a **Google Drive link**, then filter and view the **most linked-to destinations**."
)

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


def safe_str_series(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("")


def parse_gdrive_file_id(url: str) -> str | None:
    """
    Extract Google Drive file ID from common share URL formats.
    Supports:
      - https://drive.google.com/file/d/<ID>/view?...
      - https://drive.google.com/open?id=<ID>
      - https://drive.google.com/uc?id=<ID>&export=download
      - https://docs.google.com/uc?id=<ID>&export=download
    """
    if not url:
        return None
    url = url.strip()

    m = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)

    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)

    return None


def download_gdrive_file(file_id: str, out_path: str) -> None:
    """
    Stream-download a Google Drive file to out_path.
    Handles the 'virus scan too large' confirmation token.
    Requires requests.
    """
    if not HAS_REQUESTS:
        raise RuntimeError("The 'requests' package is required for Google Drive downloads.")

    session = requests.Session()
    base_url = "https://drive.google.com/uc?export=download"

    # 1st request
    resp = session.get(base_url, params={"id": file_id}, stream=True)
    resp.raise_for_status()

    # If Drive requires a confirmation token (large files), it sets a cookie like 'download_warning'
    token = None
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    if token:
        resp = session.get(base_url, params={"id": file_id, "confirm": token}, stream=True)
        resp.raise_for_status()

    # Stream to disk
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


# ---------------------------
# UI: Input source
# ---------------------------

st.sidebar.header("📥 Input")
input_mode = st.sidebar.radio(
    "Choose input method",
    options=["Upload file", "Google Drive link"],
    horizontal=False
)

uploaded_file = None
gdrive_url = None

if input_mode == "Upload file":
    uploaded_file = st.file_uploader(
        "Upload All Inlinks (.csv or .csv.gz)",
        type=["csv", "gz"]
    )
else:
    if not HAS_REQUESTS:
        st.error("To use Google Drive links, add `requests` to requirements.txt.")
        st.stop()

    gdrive_url = st.text_input(
        "Paste Google Drive share link",
        placeholder="https://drive.google.com/file/d/FILE_ID/view?usp=sharing"
    )
    st.caption("Tip: set sharing to **Anyone with the link (Viewer)** for easiest access.")

# ---------------------------
# UI: Processing settings
# ---------------------------

st.sidebar.header("⚙️ Processing")
chunksize = st.sidebar.number_input(
    "Chunk size (rows)",
    min_value=50_000,
    max_value=1_000_000,
    value=200_000,
    step=50_000,
    help="Bigger chunks = faster, but more RAM. On Streamlit Cloud, 200k is a safe default."
)

sample_rows = st.sidebar.number_input(
    "Sample rows for filter value pickers",
    min_value=10_000,
    max_value=1_000_000,
    value=200_000,
    step=50_000,
    help="We use a sample to populate multiselect options without scanning the whole file."
)

force_string_dtypes = st.sidebar.checkbox(
    "Treat all columns as text (recommended)",
    value=True
)

# Unique counting mode
st.sidebar.header("🧮 Unique counting")
unique_mode = st.sidebar.radio(
    "Unique Source Pages per Destination",
    options=[
        "Auto (best available)",
        "Approx (HyperLogLog)",
        "Exact (small files only)"
    ],
    help=(
        "Exact unique counts can blow up memory on large crawls. "
        "HyperLogLog gives very good approximations with tiny memory."
    )
)

if unique_mode == "Approx (HyperLogLog)" and not HAS_HLL:
    st.sidebar.warning("Install `datasketch` for HyperLogLog. Falling back to exact (may OOM on big files).")
if unique_mode == "Auto (best available)" and not HAS_HLL:
    st.sidebar.info("datasketch not found → Auto will use Exact unique counts (may not scale).")

# ---------------------------
# UI: Filters
# ---------------------------

st.subheader("🎛️ Filters")

colA, colB, colC = st.columns([1, 1, 2])
with colA:
    remove_self = st.checkbox("Remove self-referring links (Source == Destination)", value=False)
with colB:
    exclude_params = st.checkbox("Exclude Destination URLs with query parameters ('?')", value=False)
with colC:
    st.caption("These filters are applied during chunk processing (memory-friendly).")

st.markdown("### 🧭 Exclude breadcrumb / structural navigation (Link Path patterns)")
exclude_link_path = st.checkbox("Enable Link Path exclusions", value=False)

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
exclude_destination = st.checkbox("Enable Destination URL exclusions", value=False)

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

st.markdown("### 🧾 Column include filters (sample-based)")
st.caption(
    "Because we process in chunks, we populate these dropdowns from a **sample** of the file "
    "(first N rows). This keeps the app fast and Cloud-safe."
)

FILTER_COLUMNS = ["Type", "Follow", "Status Code", "Status", "Link Position", "Link Origin", "Target", "Rel", "Path Type"]

# Deduped ranking option (no global dedupe set needed)
use_deduped_ranking = st.checkbox(
    "Rank by 'Unique Source Pages' (deduped Source→Destination) instead of raw occurrences",
    value=False,
    help=(
        "True global deduplication of Source→Destination pairs across an entire huge file can be memory-heavy. "
        "Ranking by Unique Source Pages gives you the 'deduped' view you usually want."
    )
)

# ---------------------------
# Load file to temp path (upload or Drive)
# ---------------------------

def materialize_input_to_tempfile() -> str | None:
    """
    Save uploaded file or Drive download to a temp file, return path.
    We do this so pandas can chunk-read reliably.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp_path = tmp.name
    tmp.close()

    if input_mode == "Upload file":
        if uploaded_file is None:
            return None
        # write bytes to disk
        data = uploaded_file.getbuffer()
        with open(tmp_path, "wb") as f:
            f.write(data)
        return tmp_path

    # Google Drive mode
    if not gdrive_url:
        return None
    file_id = parse_gdrive_file_id(gdrive_url)
    if not file_id:
        st.error("Could not parse a Google Drive file ID from that link.")
        return None

    with st.spinner("Downloading from Google Drive..."):
        try:
            download_gdrive_file(file_id, tmp_path)
        except Exception as e:
            st.error(f"Drive download failed: {e}")
            return None
    return tmp_path


tmp_path = materialize_input_to_tempfile()
if tmp_path is None:
    st.info("Provide an upload or a Google Drive link to continue.")
    st.stop()

# Try to infer compression from filename contents (uploaded gz is still saved as .csv)
# pandas can infer gz from file content? not reliably. We'll allow user hint:
st.sidebar.header("🗜️ Compression")
compression_hint = st.sidebar.selectbox(
    "File compression",
    options=["infer", "gzip", "none"],
    index=0,
    help="If you uploaded a .csv.gz but it was saved without .gz extension, choose gzip."
)
compression = None if compression_hint == "none" else compression_hint

# ---------------------------
# Build sample for column filter pickers
# ---------------------------

@st.cache_data(show_spinner=False)
def read_sample(path: str, nrows: int, compression_hint: str, force_str: bool) -> pd.DataFrame:
    read_kwargs = dict(
        nrows=nrows,
        low_memory=False,
        compression=None if compression_hint == "none" else compression_hint,
    )
    if force_str:
        read_kwargs["dtype"] = "string"

    # encoding fallback
    try:
        return pd.read_csv(path, **read_kwargs)
    except UnicodeDecodeError:
        read_kwargs["encoding"] = "cp1252"
        return pd.read_csv(path, **read_kwargs)

sample_df = read_sample(tmp_path, int(sample_rows), compression_hint, force_string_dtypes)

# Validate core columns
for core in ["Source", "Destination"]:
    if core not in sample_df.columns:
        st.error(f"Missing expected column in file sample: {core}")
        st.stop()

# Build sample-based multiselects
selected_values_by_col: dict[str, set[str]] = {}
for col in FILTER_COLUMNS:
    if col in sample_df.columns:
        vals = sorted(sample_df[col].astype("string").fillna("").unique().tolist(), key=lambda x: str(x).lower())
        display_vals = ["(blank)" if str(v).strip() == "" else str(v) for v in vals]
        selected_display = st.multiselect(
            f"Include values for **{col}** (sample-based)",
            options=display_vals,
            default=display_vals
        )
        # map display -> real string
        selected_real = set("" if d == "(blank)" else d for d in selected_display)
        selected_values_by_col[col] = selected_real

# Compile pattern regexes
rx_link_path = compile_contains_patterns(link_path_patterns) if exclude_link_path else None
rx_dest = compile_contains_patterns(dest_patterns) if exclude_destination else None

# ---------------------------
# Chunk processing + aggregation
# ---------------------------

def choose_unique_strategy(mode: str) -> str:
    if mode == "Exact (small files only)":
        return "exact"
    if mode == "Approx (HyperLogLog)":
        return "hll" if HAS_HLL else "exact"
    # Auto
    return "hll" if HAS_HLL else "exact"

unique_strategy = choose_unique_strategy(unique_mode)

# Aggregates
# total occurrences per destination (raw rows after filters)
total_inlinks: dict[str, int] = {}

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

def apply_filters_to_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    # Ensure strings for operations
    # (If dtype already string, this is cheap)
    for c in ["Source", "Destination"]:
        chunk[c] = chunk[c].astype("string").fillna("")

    # Remove self-referring
    if remove_self:
        src_norm = chunk["Source"].map(normalize_url_for_compare)
        dst_norm = chunk["Destination"].map(normalize_url_for_compare)
        chunk = chunk[src_norm != dst_norm]

    # Exclude params
    if exclude_params:
        chunk = chunk[~chunk["Destination"].str.contains(r"\?", regex=True, na=False)]

    # Link Path exclusion
    if rx_link_path is not None and "Link Path" in chunk.columns:
        lp = chunk["Link Path"].astype("string").fillna("")
        chunk = chunk[~lp.str.contains(rx_link_path, na=False)]

    # Destination exclusion
    if rx_dest is not None:
        chunk = chunk[~chunk["Destination"].str.contains(rx_dest, na=False)]

    # Column include filters (from sample selections)
    for col, allowed in selected_values_by_col.items():
        if col in chunk.columns:
            s = chunk[col].astype("string").fillna("")
            chunk = chunk[s.isin(allowed)]

    return chunk

def process_file_in_chunks(path: str):
    read_kwargs = dict(
        chunksize=int(chunksize),
        low_memory=False,
        compression=None if compression_hint == "none" else compression_hint,
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
    chunks_seen = 0

    for chunk in iterator:
        chunks_seen += 1
        total_rows_seen += len(chunk)

        # Require columns exist in chunk
        if "Source" not in chunk.columns or "Destination" not in chunk.columns:
            continue

        filtered = apply_filters_to_chunk(chunk)
        if filtered.empty:
            # Update progress text only
            if chunks_seen % 5 == 0:
                progress.progress(min(0.99, chunks_seen / (chunks_seen + 50)), text=f"Processed {total_rows_seen:,} rows...")
            continue

        # Aggregate totals
        dest_series = filtered["Destination"].astype("string").fillna("")
        src_series = filtered["Source"].astype("string").fillna("")

        # Total inlink occurrences (raw rows after filters)
        counts = dest_series.value_counts(dropna=False)
        for dest, cnt in counts.items():
            dest = str(dest)
            total_inlinks[dest] = total_inlinks.get(dest, 0) + int(cnt)

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

# Run processing
run = st.button("🚀 Run analysis", type="primary")

if run:
    with st.spinner("Processing file in chunks..."):
        process_file_in_chunks(tmp_path)

    # Build output table
    destinations = list(total_inlinks.keys())

    if unique_strategy == "exact":
        assert unique_sources_exact is not None
        unique_counts = {d: len(unique_sources_exact.get(d, set())) for d in destinations}
        unique_note = "Exact"
    else:
        assert unique_sources_hll is not None
        unique_counts = {d: int(unique_sources_hll[d].count()) if d in unique_sources_hll else 0 for d in destinations}
        unique_note = "Approx (HyperLogLog++)"

    out = pd.DataFrame({
        "Destination": destinations,
        "Total_Inlink_Occurrences": [total_inlinks.get(d, 0) for d in destinations],
        "Unique_Source_Pages": [unique_counts.get(d, 0) for d in destinations],
    })

    # Ranking choice
    if use_deduped_ranking:
        out = out.sort_values(["Unique_Source_Pages", "Total_Inlink_Occurrences"], ascending=False)
        st.info(f"Ranking by **Unique_Source_Pages** (deduped view). Unique count mode: **{unique_note}**.")
    else:
        out = out.sort_values(["Total_Inlink_Occurrences", "Unique_Source_Pages"], ascending=False)
        st.info(f"Ranking by **Total_Inlink_Occurrences** (raw rows). Unique count mode: **{unique_note}**.")

    top_n = st.number_input("Rows to show", min_value=10, max_value=500, value=50, step=10)
    out_top = out.head(int(top_n))

    st.subheader("🏆 Top destinations")
    st.dataframe(out_top, use_container_width=True)

    # Downloads
    st.subheader("⬇️ Downloads")
    c1, c2 = st.columns(2)

    with c1:
        st.download_button(
            "Download top destinations CSV",
            data=out_top.to_csv(index=False).encode("utf-8"),
            file_name="top_destinations.csv",
            mime="text/csv",
        )

    with c2:
        st.download_button(
            "Download full destination summary CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="destinations_summary.csv",
            mime="text/csv",
        )

# Cleanup note
with st.expander("Notes / How to use for very large files"):
    st.markdown(
        """
- **Best practice:** upload **`.csv.gz`** (compress your CSV). All Inlinks files compress very well.
- **Google Drive mode** lets you bypass the browser upload limit, because the app downloads the file server-side.
- **Chunk processing** avoids loading the whole file into memory.
- **Unique Source Pages**:
  - If `datasketch` is installed, the app uses **HyperLogLog++** for large-file-safe approximate unique counts.
  - Exact unique counts can be memory heavy on big crawls.
"""
    )

# Best-effort temp file cleanup (optional)
# Streamlit reruns frequently; deleting immediately can break subsequent runs.
# You can leave temp files or implement periodic cleanup if deploying elsewhere.




































