import subprocess
import sys
# Ensure Playwright is installed for browser automation
subprocess.run([sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"], check=False)

import time
import json
import re
import os
import requests
import numpy as np
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import cloudscraper
import openai
from requests.exceptions import ReadTimeout
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import csv

# Optional fuzzy matching (RapidFuzz). Falls back to a simple Jaccard if unavailable.
try:
    from rapidfuzz import fuzz
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="AI Overview/AI Mode query fan-out gap analysis", layout="wide")
st.title("🔍 AI Overview/AI Mode Query Fan-Out Gap Analysis")

# --- BEFORE YOU START PANEL ---
st.markdown(
    """
    ### 🛠 Before You Start

    **What this tool does**  
    1. **Scan your content** – Analyzes each page’s headings and body copy to map your current coverage.  
    2. **Generate AI-powered queries** – Uses Google’s Gemini with Google Search grounding (the same tech behind AI Overviews & AI Mode) to create multi-layer “fan-out” queries that reflect how AI explores a topic.  
    3. **Pinpoint coverage gaps** – Compares those queries against your page to reveal exactly what’s missing.  
    4. **Score your coverage** – Calculates a coverage percentage showing how well your content meets AI-driven search intent.  
    5. **Give you ready-to-add improvements** – Outputs section ideas you can drop into your content to align with what Google’s AI actually surfaces—boosting visibility, authority, and user trust.  

    **How to run it**  
    - Paste in **one or more URLs** (one per line) into the text area.  
    - Click **Start Audit**.  
    - **Keep the browser tab active** – don’t let your device sleep or the session pause while it’s running.  

    **How long it takes**  
    - Expect around **60–120 seconds per URL** depending on page size and connection speed.  
    - Multiple URLs run sequentially, so larger batches will take proportionally longer.  

    **What you’ll get**  
    - **Detailed results** – Every query, whether it’s covered, plus explanations.  
    - **Summary view** – Overall coverage % and fan-out query stats.  
    - **Action list** – Specific sections or topics to add for stronger AI search performance.  
    - **Downloadable CSVs** – Detailed, Summary, and Actions for editing, sharing, or importing into your workflow.  
    """,
    unsafe_allow_html=True
)

# --- Fixed Configuration (all defaults, no UI controls) ---
gemini_temp       = 0.4      # Diversity for fan-out generation
gpt_temp          = 0.1      # Temperature for gap reasoning
attempts          = 1        # Gemini calls per input
candidate_count   = 7        # Default candidates per Gemini call
BODY_CHAR_LIMIT   = 2000     # Limit body text passed to GPT per batch

# Dedupe defaults
ENABLE_DEDUPE     = True
FUZZY_RATIO       = 92
EMBED_ON          = True
EMBED_THRESHOLD   = 0.86
EMBED_MODEL       = "text-embedding-3-small"
DEFAULT_USER_AGENT = "QueryFanOutBot/1.0"

# Performance defaults
MAX_WORKERS       = 6
LVL2_CANDIDATES   = 4
LVL2_TIMEOUT      = 15
LVL1_TIMEOUT      = 30
GPT_BATCH_SIZE    = 8

# Normalization default
NORMALIZE_YEAR_SUFFIX = True

# --- Load API Keys from Streamlit Secrets ---
openai.api_key   = st.secrets["openai"]["api_key"]
gemini_api_key   = st.secrets["google"]["gemini_api_key"]

def _secret_value(path, default=None):
    current = st.secrets
    for key in path:
        if key not in current:
            return default
        current = current[key]
    return current

def get_proxy_url():
    return (
        _secret_value(("proxy", "server"))
        or os.getenv("PROXY_SERVER")
        or os.getenv("HTTPS_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("HTTP_PROXY")
        or os.getenv("http_proxy")
    )

def get_proxy_username():
    return _secret_value(("proxy", "username")) or os.getenv("PROXY_USERNAME")

def get_proxy_password():
    return _secret_value(("proxy", "password")) or os.getenv("PROXY_PASSWORD")

def get_user_agent():
    return (
        _secret_value(("crawler", "user_agent"))
        or os.getenv("CRAWLER_USER_AGENT")
        or DEFAULT_USER_AGENT
    )

def get_requests_proxy_map():
    proxy_url = get_proxy_url()
    if not proxy_url:
        return None
    return {"http": proxy_url, "https": proxy_url}

def get_playwright_proxy_settings():
    proxy_url = get_proxy_url()
    if not proxy_url:
        return None
    proxy = {"server": proxy_url}
    username = get_proxy_username()
    password = get_proxy_password()
    if username:
        proxy["username"] = username
    if password:
        proxy["password"] = password
    return proxy

# --- URL Input Area ---
urls_input = st.text_area(
    "Enter one URL per line to audit:",
    placeholder="https://example.com/page1\nhttps://example.com/page2"
)

# --- Style the Start Button ---
st.markdown(
    """
    <style>
    div.stButton > button:first-child { background-color: #e63946; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Initialize Session State ---
def init_state():
    defaults = {
        "last_urls": [],
        "processed": False,
        "detailed": [],
        "summary": [],
        "actions": [],
        "skipped": [],
        "h1_fanout_cache": {},
        "fanout_layer2_cache": {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
init_state()

# --- Parse and Detect URL List ---
urls = [line.strip() for line in urls_input.splitlines() if line.strip()]

# --- Reset state when URLs change ---
if urls and st.session_state.last_urls != urls:
    st.session_state.last_urls = urls
    st.session_state.processed = False
    st.session_state.detailed.clear()
    st.session_state.summary.clear()
    st.session_state.actions.clear()
    st.session_state.skipped.clear()
    st.session_state.h1_fanout_cache.clear()
    st.session_state.fanout_layer2_cache.clear()

# =========================
# NORMALIZATION UTILITIES
# =========================

# Handles endings like: " ... 2024", " ... (2024)", " ... in 2024", " ... 2024/25", " ... 2024-2025"
YEAR_SUFFIX_RE = re.compile(r"""
    (?:\s*[\(\-]?\s*)        # optional spacing / '(' / '-' before the year
    (?:in\s+)?               # optional 'in '
    ((?:19|20)\d{2})         # base 4-digit year (capture)
    (?:\s*/\s*\d{2}          # ' /25' short range
       |-(?:19|20)\d{2}      # or '-2025' full range
    )?                       # optional year-range
    \)?\s*$                  # optional ')' then end of string
""", re.IGNORECASE | re.VERBOSE)

def strip_trailing_years(q: str, min_year: int = 2000, max_year: int | None = None) -> str:
    """Remove a trailing year or year-range only at the end of the query."""
    if max_year is None:
        max_year = datetime.now().year + 1  # allow next-year planning queries
    m = YEAR_SUFFIX_RE.search(q)
    if not m:
        return q
    try:
        y = int(m.group(1))
    except Exception:
        return q
    if min_year <= y <= max_year:
        return q[:m.start()].rstrip()
    return q

# =========================
# DEDUPE UTILITIES
# =========================

STOPWORDS = {
    "the","and","of","in","to","a","for","with","on","about",
    "vs","vs.","is","are","your","what","how","why","more",
    "latest","new","does","do","an"
}

def canonicalize(q: str) -> str:
    q = q.lower()
    q = re.sub(r"[^\w\s]", " ", q)
    if STOPWORDS:
        q = re.sub(r"\b(" + "|".join(map(re.escape, STOPWORDS)) + r")\b", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def _token_set_ratio(a: str, b: str) -> int:
    if HAVE_RAPIDFUZZ:
        return int(fuzz.token_set_ratio(a, b))
    ta, tb = set(canonicalize(a).split()), set(canonicalize(b).split())
    if not ta and not tb: return 100
    if not ta or not tb:  return 0
    jacc = len(ta & tb) / len(ta | tb)
    return int(round(jacc * 100))

def dedupe_exact(queries):
    seen = set()
    kept, groups = [], {}
    for q in queries:
        c = canonicalize(q)
        if c in seen:
            for rep in kept:
                if canonicalize(rep) == c:
                    groups[rep].append(q)
                    break
        else:
            seen.add(c)
            kept.append(q)
            groups[q] = [q]
    return kept, groups

def dedupe_token_set(queries, min_ratio=92):
    kept = []
    groups = {}
    for q in queries:
        attached = False
        for k in kept:
            score = _token_set_ratio(q, k)
            if score >= min_ratio:
                groups[k].append(q)
                attached = True
                break
        if not attached:
            kept.append(q)
            groups[q] = [q]
    return kept, groups

def get_embeddings(queries, model="text-embedding-3-small"):
    resp = openai.embeddings.create(model=model, input=queries)
    data = getattr(resp, "data", None) or resp["data"]
    out = []
    for d in data:
        emb = getattr(d, "embedding", None)
        if emb is None:
            emb = d.get("embedding", [])
        out.append(np.array(emb, dtype=float))
    return out

def cosine(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def dedupe_embeddings(queries, threshold=0.86, model="text-embedding-3-small"):
    if not queries:
        return [], {}
    try:
        vecs = get_embeddings(queries, model=model)
    except Exception as e:
        st.warning(f"Embedding dedupe skipped (embedding error): {e}")
        return queries, {q: [q] for q in queries}
    kept, groups, removed = [], {}, set()
    for i, qi in enumerate(queries):
        if i in removed: continue
        kept.append(qi)
        groups[qi] = [qi]
        for j in range(i+1, len(queries)):
            if j in removed: continue
            sim = cosine(vecs[i], vecs[j])
            if sim >= threshold:
                groups[qi].append(queries[j])
                removed.add(j)
    return kept, groups

def dedupe_pipeline(
    queries,
    use_exact=True,
    fuzzy_ratio=92,
    use_embed=True,
    embed_threshold=0.86,
    embed_model="text-embedding-3-small"
):
    if not queries:
        return [], {}
    q = queries[:]
    if use_exact:
        q, groups_exact = dedupe_exact(q)
    else:
        groups_exact = {x: [x] for x in q}
    q2, groups_ts = dedupe_token_set(q, min_ratio=fuzzy_ratio)
    merged_ts = {rep: [] for rep in q2}
    for rep in q2:
        merged_ts[rep].extend(groups_ts[rep])
    if use_embed and len(q2) > 1:
        reps, sem_groups = dedupe_embeddings(q2, threshold=embed_threshold, model=embed_model)
        final_groups = {r: [] for r in reps}
        for r in reps:
            for member in sem_groups[r]:
                final_groups[r].extend(merged_ts.get(member, [member]))
        for r in final_groups:
            seen, ded = set(), []
            for item in final_groups[r]:
                if item not in seen:
                    seen.add(item); ded.append(item)
            final_groups[r] = ded
        return reps, final_groups
    return q2, merged_ts

# =========================
# FETCH FAN-OUTS (parametrized + retries)
# =========================

def fetch_query_fan_outs_multi(text, attempts=1, temp=0.0, cand_count=None, timeout_s=60):
    """Generate fan-out queries via Gemini, with retries on ReadTimeout."""
    cand = cand_count if cand_count is not None else candidate_count
    queries = []
    for _ in range(attempts):
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={gemini_api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": temp, "candidateCount": cand},
        }
        response = None
        for retry in range(2):
            try:
                response = requests.post(endpoint, json=payload, timeout=timeout_s)
                response.raise_for_status()
                break
            except ReadTimeout:
                time.sleep(0.8)
            except Exception as e:
                st.warning(f"Fan-out fetch failed for '{text}': {e}")
                break
        if not response:
            continue
        try:
            data = response.json().get("candidates", [])
            for cand in data:
                queries.extend(cand.get("groundingMetadata", {}).get("webSearchQueries", []) or [])
        except Exception as e:
            st.warning(f"Error parsing fan-out response JSON for '{text}': {e}")
    return queries

# Parallel Level-2 expansion
def expand_level2_parallel(level1, attempts=1, temp=0.4, max_workers=6, cand_count=4, timeout_s=15):
    results = []
    if not level1:
        return results
    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(level1)))) as ex:
        futs = {
            ex.submit(fetch_query_fan_outs_multi, q, attempts, temp, cand_count, timeout_s): q
            for q in level1
        }
        for fut in as_completed(futs):
            try:
                results.extend(fut.result() or [])
            except Exception as e:
                st.warning(f"Level-2 expansion error: {e}")
    return results

# =========================
# CONTENT EXTRACTION (fast path first) + caching
# =========================

def extract_content_fast(url):
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    scraper.headers.update({"User-Agent": get_user_agent()})
    r = scraper.get(url, timeout=20, proxies=get_requests_proxy_map())
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def extract_content_playwright(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            proxy=get_playwright_proxy_settings()
        )
        context = browser.new_context(user_agent=get_user_agent())
        page = context.new_page()
        resp = page.goto(url, timeout=45000)
        if resp and resp.status == 403:
            raise RuntimeError("HTTP 403 Forbidden")
        html = page.content()
        context.close()
        browser.close()
    return BeautifulSoup(html, "html.parser")

@st.cache_data(show_spinner=False, ttl=3600)
def cached_extract(url):
    try:
        soup = extract_content_fast(url)
        if not soup.find("h1"):
            soup = extract_content_playwright(url)
    except Exception:
        soup = extract_content_playwright(url)
    h1_tag = soup.find("h1")
    h1_text = h1_tag.get_text(strip=True) if h1_tag else ""
    headings = [(tag.name.upper(), tag.get_text(strip=True)) for tag in soup.find_all(["h2","h3","h4"])]
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    list_items = [li.get_text(strip=True) for li in soup.find_all("li")]
    body = "\n".join(paragraphs + list_items)
    return h1_text, headings, body, None

@st.cache_data(show_spinner=False, ttl=86400)
def cached_fanouts(text, cand_count, temp, timeout_s):
    return fetch_query_fan_outs_multi(text, attempts=attempts, temp=temp, cand_count=cand_count, timeout_s=timeout_s)

# =========================
# PROMPT BUILDING + GPT CALL (batched)
# =========================

def build_prompt(h1, headings, body, queries):
    lines = [
        "I’m auditing this page for content gaps.",
        f"Main topic (H1): {h1}",
        "",
        "Existing Headings:",
    ]
    for lvl, txt in headings:
        lines.append(f"- {lvl}: {txt}")
    lines += ["", "Page Body Text:", body[:BODY_CHAR_LIMIT], "", "Queries to check coverage:"]
    for q in queries:
        lines.append(f"- {q}")
    lines += [
        "",
        "Please provide coverage entries for *all* of the above queries, even if covered=false.",
        "",
        "Given the above headings and body text, return ONLY a JSON array with keys:",
        "query (string), covered (true/false), explanation (string).",
        'Example: [{"query":"...","covered":true,"explanation":"..."}]'
    ]
    return "\n".join(lines)

def get_explanations(prompt, temperature=0.1, max_retries=2):
    system_msg = "You are an SEO content gap auditor. Output ONLY a JSON array."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    last_resp = ""
    for _ in range(max_retries):
        try:
            resp = openai.chat.completions.create(
                model="gpt-4o", messages=messages, temperature=temperature
            )
            text = resp.choices[0].message.content.strip()
            last_resp = text
            match = re.search(r"\[.*\]", text, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            time.sleep(0.8)
    st.warning(f"Failed to parse OpenAI response as JSON. Raw response: {last_resp}")
    return []

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def get_explanations_batched(h1, headings, body, queries, batch_size=8, temperature=0.1):
    out = []
    for group in chunked(queries, batch_size):
        prompt = build_prompt(h1, headings, body, group)
        out.extend(get_explanations(prompt, temperature=temperature))
    return out

# =========================
# GOOGLE SHEETS–FRIENDLY CSV HELPERS
# =========================

_SHEETS_RISK_PREFIXES = ("=", "+", "-", "@", "\t")

def _sanitize_for_sheets(val):
    """
    Prevent formula-injection in Google Sheets by prefixing risky leading chars with a single quote.
    Convert NaNs to empty strings. Preserve numbers.
    """
    if pd.isna(val):
        return ""
    if isinstance(val, (int, float, np.number)):
        return val
    s = str(val)
    if s.startswith(_SHEETS_RISK_PREFIXES):
        return "'" + s
    return s

def _coerce_summary_frame(rows):
    """Normalize & order summary columns for clean Sheets import."""
    cols = ["Address", "Fan-out Count (raw)", "Queries Used (after dedupe)", "Coverage (%)"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    df["Address"] = df["Address"].astype("string")
    for c in ["Fan-out Count (raw)", "Queries Used (after dedupe)", "Coverage (%)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df = df[cols].map(_sanitize_for_sheets)
    return df

def _coerce_actions_frame(rows):
    """Return a 2-col Sheets-safe frame: Address, Recommended sections to add.
       Merges legacy key casing ('Recommended Sections to Add') into the new key."""
    cols = ["Address", "Recommended sections to add"]
    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)

    legacy_col = "Recommended Sections to Add"
    new_col = "Recommended sections to add"

    if new_col not in df.columns and legacy_col in df.columns:
        df[new_col] = df[legacy_col]
    elif new_col in df.columns and legacy_col in df.columns:
        df[new_col] = df[new_col].fillna(df[legacy_col])

    if "Address" not in df.columns:
        df["Address"] = pd.NA
    if new_col not in df.columns:
        df[new_col] = ""

    df = df[["Address", new_col]]
    df = df.map(_sanitize_for_sheets)
    return df

# =========================
# MAIN AUDIT LOOP (no cloud, no sidebar)
# =========================

# Prepare URLs (from UI)
urls_ui = [u.strip() for u in urls_input.splitlines() if u.strip()]

start_clicked = st.button("Start Audit")

if start_clicked and urls_ui and not st.session_state.processed:

    progress_bar = st.progress(0)
    status_text  = st.empty()
    start_time   = time.time()

    total = len(urls_ui)

    for idx, url in enumerate(urls_ui):

        # Simple status (ETA removed)
        status_text.text(f"Processing {idx+1}/{total}")
        progress_bar.progress(int(((idx+1) / total) * 100))

        # Content extraction (cached, fast-path first)
        try:
            h1_text, headings, body, err = cached_extract(url)
        except Exception as e:
            err_str = str(e)
            if "libglib" in err_str or "BrowserType.launch" in err_str or "shared object file" in err_str:
                reason = "This page requires a browser to load but the browser engine is unavailable. Try re-deploying the app — if the problem persists, contact support."
            else:
                reason = f"Fetch error ({e})"
            st.warning(f"Skipped {url}: {reason}")
            st.session_state.skipped.append({"Address": url, "Reason": reason})
            continue

        if err or not h1_text:
            reason = err or "No H1 found"
            st.warning(f"Skipped {url}: {reason}")
            st.session_state.skipped.append({"Address": url, "Reason": reason})
            continue

        # Level 1 fan-out (cached)
        lvl1 = st.session_state.h1_fanout_cache.get(h1_text) or cached_fanouts(
            h1_text, candidate_count, gemini_temp, LVL1_TIMEOUT
        )
        st.session_state.h1_fanout_cache[h1_text] = lvl1

        # Level 2 fan-out (parallel & lighter)
        level2_key = ("__lvl2__", h1_text, LVL2_CANDIDATES, LVL2_TIMEOUT)
        level2 = st.session_state.fanout_layer2_cache.get(level2_key)
        if level2 is None:
            level2 = expand_level2_parallel(
                lvl1,
                attempts=attempts,
                temp=gemini_temp,
                max_workers=MAX_WORKERS,
                cand_count=LVL2_CANDIDATES,
                timeout_s=LVL2_TIMEOUT
            )
            st.session_state.fanout_layer2_cache[level2_key] = level2

        all_qs = (lvl1 or []) + (level2 or [])

        # Normalize queries by stripping trailing years (optional, before dedupe)
        if NORMALIZE_YEAR_SUFFIX:
            all_qs = [strip_trailing_years(q) for q in all_qs]
            all_qs = [q for q in all_qs if q]

        if not all_qs:
            reason = "No queries generated"
            st.warning(f"Skipped {url}: {reason}")
            st.session_state.skipped.append({"Address": url, "Reason": reason})
            continue

        # --- Dedupe (defaults) ---
        raw_queries = all_qs[:]  # keep a copy for transparency
        if ENABLE_DEDUPE:
            reps, groups = dedupe_pipeline(
                all_qs,
                use_exact=True,
                fuzzy_ratio=FUZZY_RATIO,
                use_embed=EMBED_ON,
                embed_threshold=EMBED_THRESHOLD,
                embed_model=EMBED_MODEL
            )
            queries_for_prompt = reps
            grouped_view = {rep: "; ".join(members) for rep, members in groups.items()}
        else:
            queries_for_prompt = raw_queries
            grouped_view = {q: q for q in raw_queries}

        # Build prompt & call GPT (batched)
        results = get_explanations_batched(
            h1_text, headings, body, queries_for_prompt, batch_size=GPT_BATCH_SIZE, temperature=gpt_temp
        )
        if not results:
            reason = "No usable output from OpenAI"
            st.warning(f"Skipped {url}: {reason}")
            st.session_state.skipped.append({"Address": url, "Reason": reason})
            continue

        # Summaries & Actions
        covered = sum(1 for r in results if r.get("covered"))
        pct = int((covered / len(results)) * 100) if results else 0
        summary_row = {
            "Address": url,
            "Fan-out Count (raw)": len(raw_queries),
            "Queries Used (after dedupe)": len(queries_for_prompt),
            "Coverage (%)": pct
        }
        st.session_state.summary.append(summary_row)

        # Build "Actions" row with EXACTLY two columns, comma-separated list in one cell
        missing = [r.get("query") for r in results if not r.get("covered")]
        actions_row = {
            "Address": url,
            "Recommended sections to add": ", ".join(missing)  # single cell, comma-separated
        }
        st.session_state.actions.append(actions_row)

        # Detailed row
        row = {
            "Address": url,
            "H1": h1_text,
            "Headings": " | ".join(f"{l}:{t}" for l, t in headings),
            "All Queries (raw)": "; ".join(raw_queries),
            "Queries Used (final)": "; ".join(queries_for_prompt),
        }
        if ENABLE_DEDUPE:
            row["Dedupe Groups"] = " || ".join([f"{rep} => {members}" for rep, members in grouped_view.items()])
        for i, r in enumerate(results, start=1):
            row[f"Query {i}"]       = r.get("query")
            row[f"Covered {i}"]     = r.get("covered")
            row[f"Explanation {i}"] = r.get("explanation")
        st.session_state.detailed.append(row)

    st.session_state.processed = True

# --- Display / Download Results ---
if st.session_state.processed:
    st.header("Results")

    if st.session_state.detailed:
        st.subheader("Detailed")
        df = pd.DataFrame(st.session_state.detailed)
        st.download_button(
            "Download Detailed CSV",
            df.to_csv(index=False).encode("utf-8"),
            "detailed.csv",
            "text/csv"
        )
        st.dataframe(df)

    if st.session_state.summary:
        st.subheader("Summary")
        df_summary = _coerce_summary_frame(st.session_state.summary)

        # CSV tuned for Google Sheets (UTF-8, comma delimiter, Unix newlines, safe cell sanitization)
        csv_bytes = df_summary.to_csv(
            index=False,
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL
        ).encode("utf-8")

        st.download_button(
            "Download Summary CSV (Google Sheets)",
            csv_bytes,
            file_name="summary.csv",
            mime="text/csv"
        )

        st.dataframe(df_summary)

    if st.session_state.actions:
        st.subheader("Actions")
        df_actions = _coerce_actions_frame(st.session_state.actions)

        st.download_button(
            "Download Actions CSV (Google Sheets)",
            df_actions.to_csv(index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL).encode("utf-8"),
            "actions.csv",
            "text/csv"
        )

        st.dataframe(df_actions)

    if st.session_state.skipped:
        st.subheader("Skipped URLs & Reasons")
        st.table(pd.DataFrame(st.session_state.skipped))






















