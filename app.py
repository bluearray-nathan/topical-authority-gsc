import subprocess
import sys
# Ensure Playwright is installed for browser automation
tools_install = subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=False)

import time
import json
import re
import requests
import numpy as np
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import cloudscraper
import openai
from requests.exceptions import ReadTimeout

# Optional fuzzy matching (RapidFuzz). Falls back to a simple Jaccard if unavailable.
try:
    from rapidfuzz import fuzz
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="AI Overview/AI Mode query fan-out gap analysis", layout="wide")
st.title("🔍 AI Overview/AI Mode Query Fan-Out Gap Analysis")

st.sidebar.header("About this tool")
st.sidebar.write(
    """
This tool identifies content gaps by:
1. Extracting your page's H1 headings and body text.
2. Generating user query fan-outs for each H1 query using Google Gemini.
3. Performing a second-level fan-out on those queries to maximize coverage.
4. Comparing all generated queries against the page's full content to highlight missing topics.
"""
)

# --- Fixed Configuration ---
gemini_temp = 0.4      # Diversity for fan-out generation
gpt_temp    = 0.1      # Temperature for gap reasoning
attempts    = 1        # Number of Gemini aggregation calls
candidate_count = 7    # Number of candidates per call
BODY_CHAR_LIMIT = 2000 # Send only first N chars of body text to keep tokens sane

# --- Dedupe Controls (Sidebar) ---
st.sidebar.subheader("Dedupe Options")
enable_dedupe = st.sidebar.checkbox("Enable dedupe", value=True)
fuzzy_ratio   = st.sidebar.slider("Fuzzy token-set threshold", 80, 100, 92)
embed_on      = st.sidebar.checkbox("Use embedding dedupe", value=True)
embed_thr_pct = st.sidebar.slider("Embedding cosine threshold (%)", 70, 99, 86)
embed_threshold = embed_thr_pct / 100.0
embed_model   = st.sidebar.selectbox("Embedding model", ["text-embedding-3-small", "text-embedding-3-large"], index=0)

# --- Load API Keys from Streamlit Secrets ---
openai.api_key = st.secrets["openai"]["api_key"]
gemini_api_key = st.secrets["google"]["gemini_api_key"]

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
total = len(urls)
if total > 0:
    st.write(f"Found {total} URLs to process.")

# --- Reset State When URLs Change ---
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
# DEDUPE UTILITIES
# =========================

STOPWORDS = {
    "the","and","of","in","to","a","for","with","on","about",
    "vs","vs.","is","are","your","what","how","why","more",
    "latest","new","does","do","an"
}

def canonicalize(q: str) -> str:
    q = q.lower()
    q = re.sub(r"[^\w\s]", " ", q)  # punctuation -> space
    # remove common stopwords
    if STOPWORDS:
        q = re.sub(r"\b(" + "|".join(map(re.escape, STOPWORDS)) + r")\b", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def _token_set_ratio(a: str, b: str) -> int:
    if HAVE_RAPIDFUZZ:
        return int(fuzz.token_set_ratio(a, b))
    # Fallback: simple Jaccard on token sets scaled to 0..100
    ta, tb = set(canonicalize(a).split()), set(canonicalize(b).split())
    if not ta and not tb:
        return 100
    if not ta or not tb:
        return 0
    jacc = len(ta & tb) / len(ta | tb)
    return int(round(jacc * 100))

def dedupe_exact(queries):
    seen = set()
    kept, groups = [], {}
    for q in queries:
        c = canonicalize(q)
        if c in seen:
            # attach to first representative for that canonical form
            # find rep already stored with that canonical form
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
    # Support both SDK styles (attribute or dict-like)
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
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def dedupe_embeddings(queries, threshold=0.86, model="text-embedding-3-small"):
    if not queries:
        return [], {}
    try:
        vecs = get_embeddings(queries, model=model)
    except Exception as e:
        st.warning(f"Embedding dedupe skipped (embedding error): {e}")
        return queries, {q: [q] for q in queries}
    kept = []
    groups = {}
    removed = set()
    for i, qi in enumerate(queries):
        if i in removed:
            continue
        kept.append(qi)
        groups[qi] = [qi]
        for j in range(i+1, len(queries)):
            if j in removed:
                continue
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
    merged_groups = {q_: [q_] for q_ in q}

    if use_exact:
        q, groups = dedupe_exact(q)
        merged_groups = groups  # reset to exact-group view

    q2, groups2 = dedupe_token_set(q, min_ratio=fuzzy_ratio)
    # merge exact groups into token-set representatives
    merged_groups_ts = {rep: [] for rep in q2}
    for rep in q2:
        merged_groups_ts[rep].extend(groups2[rep])
    # Rebuild: ensure each original item is in some rep
    # (groups2 already handles it)

    if use_embed and len(q2) > 1:
        reps, sem_groups = dedupe_embeddings(q2, threshold=embed_threshold, model=embed_model)
        final_groups = {r: [] for r in reps}
        for r in reps:
            for member in sem_groups[r]:
                final_groups[r].extend(merged_groups_ts.get(member, [member]))
        # unique + preserve order
        for r in final_groups:
            seen = set()
            deduped = []
            for item in final_groups[r]:
                if item not in seen:
                    seen.add(item)
                    deduped.append(item)
            final_groups[r] = deduped
        return reps, final_groups

    return q2, merged_groups_ts

# =========================
# FETCH FAN-OUTS
# =========================

def fetch_query_fan_outs_multi(text, attempts=1, temp=0.0):
    """
    Generate fan-out queries via Gemini, with retries on ReadTimeout.
    """
    queries = []
    for attempt_i in range(attempts):
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={gemini_api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": temp, "candidateCount": candidate_count},
        }
        # Retry loop for transient read timeouts
        response = None
        for retry in range(3):
            try:
                response = requests.post(endpoint, json=payload, timeout=60)
                response.raise_for_status()
                break
            except ReadTimeout:
                st.warning(f"ReadTimeout on fan-out fetch for '{text}' (attempt {retry+1}/3), retrying...")
                time.sleep(1)
                continue
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

# =========================
# CONTENT EXTRACTION
# =========================

def extract_content(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            resp = page.goto(url, timeout=60000)
            if resp and resp.status == 403:
                return "", [], "", "HTTP 403 Forbidden"
            html = page.content()
            browser.close()
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        try:
            scraper = cloudscraper.create_scraper(
                browser={"browser":"chrome","platform":"windows","mobile":False}
            )
            r = scraper.get(url, timeout=60)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
        except Exception:
            return "", [], "", "Fetch error"

    # H1 and subheadings
    h1_tag = soup.find("h1")
    h1_text = h1_tag.get_text(strip=True) if h1_tag else ""
    headings = [(tag.name.upper(), tag.get_text(strip=True)) for tag in soup.find_all(["h2","h3","h4"])]

    # Body text: paragraphs and list items
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    list_items = [li.get_text(strip=True) for li in soup.find_all("li")]
    body = "\n".join(paragraphs + list_items)

    return h1_text, headings, body, None

# =========================
# PROMPT BUILDING + GPT CALL
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
        "Example: [{\"query\":\"...\",\"covered\":true,\"explanation\":\"...\"}]"
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
            time.sleep(1)
    st.warning(f"Failed to parse OpenAI response as JSON. Raw response: {last_resp}")
    return []

# =========================
# MAIN AUDIT LOOP
# =========================

if st.button("Start Audit") and urls and not st.session_state.processed:
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()

    # Clear previous results
    st.session_state.detailed.clear()
    st.session_state.summary.clear()
    st.session_state.actions.clear()
    st.session_state.skipped.clear()

    for idx, url in enumerate(urls, start=1):
        elapsed = time.time() - start_time
        eta = int((elapsed/idx) * (total-idx))
        status_text.text(f"Processing {idx}/{total} — ETA: {eta}s")
        progress_bar.progress(int(idx/total*100))

        h1_text, headings, body, err = extract_content(url)
        if err or not h1_text:
            reason = err or "No H1 found"
            st.warning(f"Skipped {url}: {reason}")
            st.session_state.skipped.append({"Address": url, "Reason": reason})
            continue

        # Level 1 fan-out
        lvl1 = st.session_state.h1_fanout_cache.get(h1_text) or fetch_query_fan_outs_multi(h1_text, attempts, gemini_temp)
        st.session_state.h1_fanout_cache[h1_text] = lvl1

        # Level 2 fan-out
        all_qs = []
        for q in lvl1:
            sub = st.session_state.fanout_layer2_cache.get(q) or fetch_query_fan_outs_multi(q, attempts, gemini_temp)
            st.session_state.fanout_layer2_cache[q] = sub
            all_qs.extend(sub)
        all_qs = lvl1 + all_qs

        if not all_qs:
            st.warning(f"Skipped {url}: no queries generated.")
            st.session_state.skipped.append({"Address": url, "Reason": "No queries generated"})
            continue

        # --- Dedupe (optional) ---
        raw_queries = all_qs[:]  # keep a copy for transparency
        if enable_dedupe:
            reps, groups = dedupe_pipeline(
                all_qs,
                use_exact=True,
                fuzzy_ratio=fuzzy_ratio,
                use_embed=embed_on,
                embed_threshold=embed_threshold,
                embed_model=embed_model
            )
            queries_for_prompt = reps
            grouped_view = {rep: "; ".join(members) for rep, members in groups.items()}
        else:
            queries_for_prompt = raw_queries
            grouped_view = {q: q for q in raw_queries}

        # Build prompt & call GPT
        prompt = build_prompt(h1_text, headings, body, queries_for_prompt)
        results = get_explanations(prompt, temperature=gpt_temp)
        if not results:
            st.warning(f"Skipped {url}: OpenAI returned no usable output.")
            st.session_state.skipped.append({"Address": url, "Reason": "No usable output from OpenAI"})
            continue

        # Summaries & Actions
        covered = sum(1 for r in results if r.get("covered"))
        pct = int((covered / len(results)) * 100) if results else 0
        st.session_state.summary.append({
            "Address": url,
            "Fan-out Count (raw)": len(raw_queries),
            "Queries Used (after dedupe)" : len(queries_for_prompt),
            "Coverage (%)": pct
        })
        missing = [r.get("query") for r in results if not r.get("covered")]
        st.session_state.actions.append({"Address": url, "Recommended Sections to Add": "; ".join(missing)})

        # Detailed row
        row = {
            "Address": url,
            "H1": h1_text,
            "Headings": " | ".join(f"{l}:{t}" for l, t in headings),
            "All Queries (raw)": "; ".join(raw_queries),
            "Queries Used (final)": "; ".join(queries_for_prompt),
        }
        # Include grouped variants for transparency (only when dedupe enabled)
        if enable_dedupe:
            row["Dedupe Groups"] = " || ".join([f"{rep} => {members}" for rep, members in grouped_view.items()])

        for i, r in enumerate(results, start=1):
            row[f"Query {i}"]       = r.get("query")
            row[f"Covered {i}"]     = r.get("covered")
            row[f"Explanation {i}"] = r.get("explanation")
        st.session_state.detailed.append(row)

    progress_bar.progress(100)
    status_text.text("Audit Complete!")
    st.session_state.processed = True

# --- Display / Download Results ---
if st.session_state.processed:
    st.header("Results")
    if st.session_state.detailed:
        st.subheader("Detailed")
        df = pd.DataFrame(st.session_state.detailed)
        st.download_button("Download Detailed CSV", df.to_csv(index=False).encode("utf-8"), "detailed.csv", "text/csv")
        st.dataframe(df)
    if st.session_state.summary:
        st.subheader("Summary")
        df = pd.DataFrame(st.session_state.summary)
        st.download_button("Download Summary CSV", df.to_csv(index=False).encode("utf-8"), "summary.csv", "text/csv")
        st.dataframe(df)
    if st.session_state.actions:
        st.subheader("Actions")
        df = pd.DataFrame(st.session_state.actions)
        st.download_button("Download Actions CSV", df.to_csv(index=False).encode("utf-8"), "actions.csv", "text/csv")
        st.dataframe(df)
    if st.session_state.skipped:
        st.subheader("Skipped URLs & Reasons")
        st.table(pd.DataFrame(st.session_state.skipped))













