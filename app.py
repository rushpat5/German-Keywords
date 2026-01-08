import json
import random
import time
import logging
from typing import Optional, List, Dict

import pandas as pd
import requests
import streamlit as st
import torch
from groq import Groq
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# STREAMLIT UI THEME IMPROVED
# ---------------------------------------------------------
st.set_page_config(
    page_title="Keyword Planner",
    layout="wide",
    page_icon="🇩🇪"
)

st.markdown(
    """
    <style>
        :root {
            --brand: #2b6cb0;
            --brand-light: #d9eafe;
            --bg: #ffffff;
            --text: #1a202c;
            --border: #e2e8f0;
        }

        .stApp {
            background-color: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            padding: 0;
        }

        section[data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid var(--border);
        }

        .metric-box {
            background: var(--brand-light);
            padding: 12px 18px;
            border-radius: 8px;
            border: 1px solid var(--brand);
            font-size: 1.1rem;
            font-weight: 600;
            text-align: center;
            color: var(--brand);
        }

        .context-box {
            background: #e6fffa;
            border-left: 5px solid #38b2ac;
            padding: 16px;
            border-radius: 6px;
            margin-bottom: 12px;
            color: #234e52;
            font-size: 1rem;
            line-height: 1.5;
        }

        .stButton > button {
            background-color: var(--brand) !important;
            color: white !important;
            border-radius: 6px !important;
            font-size: 1rem !important;
            padding: 8px 18px !important;
            border: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "data_processed" not in st.session_state:
    st.session_state.data_processed = False
    st.session_state.df_results = None
    st.session_state.synonyms = []
    st.session_state.strategy_text = ""
    st.session_state.working_groq_model = None
    st.session_state.current_topic = ""

# ---------------------------------------------------------
# EMBEDDING MODEL
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model(hf_token: Optional[str]):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not hf_token:
        logger.warning("Missing HF token. Falling back to MiniLM.")
        return SentenceTransformer("all-MiniLM-L6-v2").to(device)

    try:
        # Depending on sentence-transformers version, this may need use_auth_token instead of token.
        return SentenceTransformer("google/embeddinggemma-300m", token=hf_token).to(device)
    except Exception as e:
        logger.warning(f"Failed loading Gemma ({e}). Using MiniLM fallback.")
        return SentenceTransformer("all-MiniLM-L6-v2").to(device)

# ---------------------------------------------------------
# GROQ CLIENT (CACHED)
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)

# ---------------------------------------------------------
# GROQ WRAPPER WITH LIMITED RETRIES
# ---------------------------------------------------------
def run_groq(api_key: str, prompt: str, max_retries_per_model: int = 2) -> Dict:
    """
    Call Groq chat.completions with:
    - model fallback list
    - limited retries on 429
    Returns parsed JSON or {"error": "..."}.
    """
    client = get_groq_client(api_key)

    cand_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.2-90b-vision-preview",
        "llama-3.1-8b-instant"
    ]

    # If we have a known-good model from this session, try it first
    if st.session_state.working_groq_model:
        cand_models.insert(0, st.session_state.working_groq_model)

    last_error = None

    for m in cand_models:
        retries = 0
        while retries <= max_retries_per_model:
            try:
                resp = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "Return ONLY valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    model=m,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                st.session_state.working_groq_model = m
                content = resp.choices[0].message.content
                return json.loads(content)
            except Exception as e:
                err_str = str(e)
                last_error = err_str
                logger.warning(f"Groq error on model {m}: {err_str}")

                # invalid key -> fail fast
                if "401" in err_str:
                    return {"error": "INVALID_KEY"}

                # Rate limit -> backoff & retry a few times
                if "429" in err_str:
                    retries += 1
                    sleep_time = 2 * retries  # simple linear backoff
                    logger.info(f"Rate limited. Retry {retries}/{max_retries_per_model} on {m} in {sleep_time}s.")
                    time.sleep(sleep_time)
                    continue

                # Other errors -> give up on this model
                break

    return {"error": f"All models failed. Last error: {last_error}"}

# ---------------------------------------------------------
# CULTURAL TRANSLATION (GERMAN SYNONYMS + ENGLISH EXPLANATION)
# ---------------------------------------------------------
def get_cultural_translation(api_key: str, keyword: str) -> Dict:
    prompt = f"""
    Act as a German SEO expert.

    English concept: "{keyword}"

    Produce:
    - 3 short, high-quality German search terms.
    - Explanation in ENGLISH ONLY.

    The explanation MUST be English. Do not output German in explanation.

    Return STRICT JSON:
    {{
        "synonyms": ["term1", "term2", "term3"],
        "explanation": "English explanation here."
    }}
    """

    result = run_groq(api_key, prompt)
    if "error" in result:
        return result

    syns = result.get("synonyms", [])
    if isinstance(syns, str):
        syns = [syns]

    result["synonyms"] = [s.strip() for s in syns if isinstance(s, str)]
    result["explanation"] = str(result.get("explanation", "")).strip()

    return result

# ---------------------------------------------------------
# AUTOCOMPLETE MINING
# ---------------------------------------------------------
def fetch_suggestions(q: str) -> List[str]:
    url = f"https://www.google.com/complete/search?client=chrome&q={q}&hl=de&gl=de"
    try:
        # Small random delay to be polite
        time.sleep(random.uniform(0.12, 0.28))
        r = requests.get(url, timeout=2.2)
        r.raise_for_status()
        data = r.json()
        return [x for x in data[1] if isinstance(x, str)]
    except Exception as e:
        logger.warning(f"Suggestion fetch failed for '{q}': {e}")
        return []

def deep_mine(synonyms: List[str]) -> pd.DataFrame:
    modifiers = [
    "",                       # keep original seed term
    " was ist",              # what is
    " wie funktioniert",     # how does … work
    " wie macht man",        # how to
    " warum",                # why
    " tipps für",           # tips for (informational)
    " häufige fragen",      # frequently asked questions
    " problem lösen",       # solve problem
    " erklärung",           # explanation
]

    rows = []
    total = max(len(synonyms) * len(modifiers), 1)
    step = 0

    prog = st.progress(0, "Mining Google Autocomplete...")

    for s in synonyms:
        for m in modifiers:
            step += 1
            prog.progress(step / total)
            query = f"{s}{m}"
            results = fetch_suggestions(query)
            for r in results:
                rows.append({"German Keyword": r, "Seed": s})

    prog.empty()

    if not rows:
        return pd.DataFrame(columns=["German Keyword", "Seed"])

    df = pd.DataFrame(rows).drop_duplicates(subset=["German Keyword"])
    return df.reset_index(drop=True)

# ---------------------------------------------------------
# SEMANTIC RELEVANCE FILTER
# ---------------------------------------------------------
def process_keywords(df: pd.DataFrame, seeds: List[str], threshold: float, hf_token: Optional[str]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    model = load_embedding_model(hf_token)

    candidates = df["German Keyword"].tolist()
    seed_terms = list(seeds)

    topic = st.session_state.current_topic
    if topic:
        seed_terms.append(topic)

    try:
        seed_vecs = model.encode(seed_terms, prompt_name="STS", normalize_embeddings=True)
        cand_vecs = model.encode(candidates, prompt_name="STS", normalize_embeddings=True)
    except TypeError:
        seed_vecs = model.encode(seed_terms, normalize_embeddings=True)
        cand_vecs = model.encode(candidates, normalize_embeddings=True)

    sim = util.cos_sim(cand_vecs, seed_vecs)
    max_sim, _ = torch.max(sim, dim=1)

    out_df = df.copy()
    out_df["Relevance"] = max_sim.cpu().numpy()

    out_df = out_df[out_df["Relevance"] >= threshold].sort_values("Relevance", ascending=False)
    return out_df if not out_df.empty else None

# ---------------------------------------------------------
# BATCH TRANSLATION VIA GROQ (REDUCED API CALLS)
# ---------------------------------------------------------
def batch_translate_keywords(api_key: str, keywords: List[str]) -> Dict[str, str]:
    """
    Translate many German keywords to English in as few Groq calls as possible.
    Returns {german: english}.
    """
    if not keywords:
        return {}

    # Deduplicate while preserving order
    unique_keywords = list(dict.fromkeys([k for k in keywords if isinstance(k, str) and k.strip()]))
    translations: Dict[str, str] = {}

    # Chunk to avoid overly long prompts
    chunk_size = 30
    for i in range(0, len(unique_keywords), chunk_size):
        chunk = unique_keywords[i:i + chunk_size]

        prompt = f"""
        You are a professional translator.

        Translate the following German keywords into literal English (short, neutral phrases).

        Return STRICT JSON of the form:
        {{
            "translations": {{
                "german_keyword_1": "english translation 1",
                "german_keyword_2": "english translation 2"
            }}
        }}

        Keywords (JSON array):
        {json.dumps(chunk, ensure_ascii=False)}
        """

        res = run_groq(api_key, prompt)
        if "error" in res:
            logger.warning(f"Translation batch failed: {res['error']}")
            continue

        trans_block = res.get("translations", {})
        if isinstance(trans_block, dict):
            for g, e in trans_block.items():
                if isinstance(g, str) and isinstance(e, str):
                    translations[g] = e.strip() or "-"

        # Small pause between batches to be gentle on rate limits
        time.sleep(0.7)

    return translations

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.header("Engine Config")
    api_key = st.text_input("Groq API Key", type="password")
    hf_token = st.text_input("Hugging Face Token", type="password")
    threshold = st.slider("Relevance Threshold", 0.0, 1.0, 0.45, 0.05)

# ---------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------
st.title("Keyword Planner 🇩🇪")
st.markdown("#### Semantic Keyword Discovery")

keyword = st.text_input("Enter English Topic")
run_btn = st.button("Generate Keywords")

# ---------------------------------------------------------
# PIPELINE EXECUTION
# ---------------------------------------------------------
if run_btn:
    if not keyword.strip() or not api_key.strip() or not hf_token.strip():
        st.error("Please provide topic, Groq API key, and HF token.")
        st.stop()

    st.session_state.current_topic = keyword.strip()
    st.session_state.data_processed = False
    st.session_state.df_results = None

    with st.spinner("Loading embedding model…"):
        _ = load_embedding_model(hf_token)

    # 1. German synonyms + explanation
    with st.spinner("Generating German synonyms…"):
        strat = get_cultural_translation(api_key, keyword)
        if "error" in strat:
            msg = strat["error"]
            if msg == "INVALID_KEY":
                st.error("Invalid Groq API key.")
            else:
                st.error(f"Groq error: {msg}")
            st.stop()

        st.session_state.synonyms = strat.get("synonyms", [])
        st.session_state.strategy_text = strat.get("explanation", "")

    # 2. Google autocomplete mining
    df_raw = deep_mine(st.session_state.synonyms)
    if df_raw.empty:
        st.warning("No keywords found from Google autocomplete.")
        st.stop()

    # 3. Semantic filtering
    with st.spinner("Filtering by semantic relevance…"):
        df_filtered = process_keywords(df_raw, st.session_state.synonyms, threshold, hf_token)
        if df_filtered is None or df_filtered.empty:
            st.warning("No relevant keywords above the threshold.")
            st.stop()

    # 4. Batch translation (Groq-friendly)
    with st.spinner("Translating keywords…"):
        german_keywords = df_filtered["German Keyword"].tolist()
        translations_map = batch_translate_keywords(api_key, german_keywords)
        df_filtered["English"] = df_filtered["German Keyword"].map(
            lambda k: translations_map.get(k, "-")
        )

    st.session_state.df_results = df_filtered
    st.session_state.data_processed = True

# ---------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------
if st.session_state.data_processed and st.session_state.df_results is not None:
    if st.session_state.strategy_text:
        st.markdown(
            f"<div class='context-box'>{st.session_state.strategy_text}</div>",
            unsafe_allow_html=True
        )

    if st.session_state.synonyms:
        cols = st.columns(len(st.session_state.synonyms))
        for i, syn in enumerate(st.session_state.synonyms):
            cols[i].markdown(
                f"<div class='metric-box'>{syn}</div>",
                unsafe_allow_html=True
            )

    df = st.session_state.df_results
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button("📥 Download CSV", csv, "keywords.csv", "text/csv")

    st.dataframe(
        df[["German Keyword", "English", "Relevance"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Relevance": st.column_config.ProgressColumn(
                "Score",
                format="%.2f",
                min_value=0,
                max_value=1
            )
        }
    )
