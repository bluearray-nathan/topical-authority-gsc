import subprocess
import sys
# Install Playwright browser binaries if possible (non-fatal)
subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=False)

import streamlit as st
import pandas as pd
import requests
import re
import json
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import cloudscraper
import openai

# --- Streamlit UI ---
st.set_page_config(page_title="Content Gap Audit", layout="wide")
st.title("🔍 Content Gap Audit Tool")

# Load API keys from Streamlit secrets
openai.api_key      = st.secrets["openai"]["api_key"]
gemini_api_key      = st.secrets["google"]["gemini_api_key"]

# Helper: extract H1 + headings via Playwright with fallback to cloudscraper
def extract_h1_and_headings(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=60000)
            html = page.content()
            browser.close()
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        st.warning(f"Playwright failed for {url}: {e}")
        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        try:
            resp = scraper.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception as e2:
            st.warning(f"Scraper fallback failed for {url}: {e2}")
            return "", []
    h1 = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""
    headings = [(tag.name.upper(), tag.get_text(strip=True)) for tag in soup.find_all(['h2','h3','h4'])]
    return h1, headings

# Helper: fetch query fan‑outs via Google Gemini
def fetch_query_fan_outs(h1_text):
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_api_key}"
    payload = {"contents": [{"parts": [{"text": h1_text}]}], "tools": [{"google_search": {}}], "generationConfig": {"temperature": 1.0}}
    try:
        r = requests.post(endpoint, json=payload, timeout=30)
        r.raise_for_status()
        cand = r.json().get("candidates", [{}])[0]
        return cand.get("groundingMetadata", {}).get("webSearchQueries", [])
    except Exception as e:
        st.warning(f"Fan‑out fetch failed: {e}")
        return []

# Helper: build GPT prompt for JSON output
def build_prompt(h1, headings, queries):
    lines = ["I’m auditing this page for content gaps.", f"Main topic (H1): {h1}", "", "1) Existing headings (in order):"]
    lines += [f"{lvl}: {txt}" for lvl, txt in headings]
    lines += ["", "2) User queries to cover:"]
    lines += [f"- {q}" for q in queries]
    lines += ["", "3) Return a JSON array with keys: query, covered (true/false), explanation.", "Example: [{\"query\":\"...\",\"covered\":true,\"explanation\":\"...\"}]"]
    return "\n".join(lines)

# Helper: call OpenAI and parse JSON
def get_explanations(prompt):
    resp = openai.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}], temperature=0.2)
    txt = resp.choices[0].message.content.strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    try:
        arr = json.loads(txt)
        return arr if isinstance(arr, list) else []
    except:
        return []

# Main: upload and process CSV
df_file = st.file_uploader("Upload Screaming Frog CSV", type=["csv"])
if df_file:
    df = pd.read_csv(df_file)
    urls = df['Address'].dropna().unique()
    total = len(urls)
    st.write(f"Found {total} URLs to process.")
    # Initialize progress bar and timer
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()

    # Detailed output rows and summary
    detailed = []
    summary = []

    for idx, url in enumerate(urls):
        # Update progress and ETA
        elapsed = time.time() - start_time
        avg_time = elapsed / (idx + 1)
        remaining = total - (idx + 1)
        eta = int(remaining * avg_time)
        progress = int((idx + 1) / total * 100)
        progress_bar.progress(progress)
        status_text.text(f"Processing {idx+1}/{total}. ETA: {eta}s")

        # Process URL
        st.write(f"Processing: {url}")
        h1, headings = extract_h1_and_headings(url)
        queries = fetch_query_fan_outs(h1)
        if not queries:
            continue
        prompt = build_prompt(h1, headings, queries)
        results = get_explanations(prompt)

        covered_count = sum(1 for it in results if it.get("covered"))
        total_q = len(results)
        pct = round((covered_count/total_q)*100) if total_q>0 else 0
        summary.append({"Address": url, "Coverage (%)": pct})

        row = {"Address": url, "H1-1": h1, "Content Structure": " | ".join(f"{lvl}:{txt}" for lvl, txt in headings)}
        for i, it in enumerate(results):
            row[f"Query {i+1}"] = it.get("query", "")
            row[f"Query {i+1} Covered"] = "Yes" if it.get("covered") else "No"
            row[f"Query {i+1} Explanation"] = it.get("explanation", "")
        detailed.append(row)

        # Finalize progress
    progress_bar.progress(100)
    status_text.text("Complete!")

    # Show and download detailed CSV
    df_det = pd.DataFrame(detailed)
    st.download_button(
        "Download Detailed CSV",
        df_det.to_csv(index=False).encode('utf-8'),
        'detailed.csv',
        'text/csv'
    )
    st.dataframe(df_det)

    # Show and download summary CSV
    df_sum = pd.DataFrame(summary)
    st.download_button(
        "Download Summary CSV",
        df_sum.to_csv(index=False).encode('utf-8'),
        'summary.csv',
        'text/csv'
    )
    st.dataframe(df_sum)







