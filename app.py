# app.py — Minimal Streamlit QA app (CPU-only, Ollama generator, FAISS retrieval on forecast SKUs)

import os, requests, numpy as np, pandas as pd, streamlit as st
import faiss
from sentence_transformers import SentenceTransformer

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
INDEX_PATH   = os.getenv("INDEX_PATH", "faiss_forecast.index")
META_PATH    = os.getenv("META_PATH",  "meta_forecast.parquet")

# ---------- Cache heavy resources ----------
@st.cache_resource(show_spinner=False)
def load_index_and_meta():
    index = faiss.read_index(INDEX_PATH)
    meta  = pd.read_parquet(META_PATH)
    return index, meta

@st.cache_resource(show_spinner=False)
def load_encoder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

index, meta = load_index_and_meta()
encoder     = load_encoder()

# ---------- Retrieval ----------
def retrieve(query: str, k: int = 3) -> pd.DataFrame:
    q = encoder.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")
    scores, idx = index.search(q, k)
    idx, scores = idx[0].tolist(), scores[0].tolist()
    out = meta.iloc[idx].copy()
    out["score"] = scores
    out["CIT"] = [f"CIT{i+1}" for i in range(len(out))]
    # truncate each chunk to ~800 chars to keep prompt small
    out["text"] = out["text"].str.slice(0, 800)
    return out.reset_index(drop=True)

# ---------- Prompting ----------
def build_prompt(question: str, rows: pd.DataFrame) -> str: # type: ignore
    ctx_lines = []
    for i, r in rows.iterrows():
        cite = f"[{r['CIT']}: {r.get('item_code','NA')} | {r.get('site','NA')} | {r.get('month_key','NA')}]"
        ctx_lines.append(f"{cite} {r['text']}")
    context = "\n".join(ctx_lines)
    return (
        "SYSTEM: Answer ONLY from CONTEXT. If info is missing, say 'Not available'. "
        "Be concise and include citation IDs (e.g., [CIT1]).\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"
    )

def ask_ollama(prompt: str, model: str = OLLAMA_MODEL, temperature: float = 0.2, max_tokens: int = 250) -> str:
    import requests
    payload = {
        "model": model,
        "prompt": prompt[:8000],          # hard cap prompt size
        "temperature": temperature,
        "num_predict": max_tokens,
        "stream": False,
        "options": {
            "num_ctx": 2048,              # keep context modest
            "num_thread": 0               # let ollama auto-pick threads
        }
    }
    r = requests.post("http://127.0.0.1:11434/api/generate", json=payload, timeout=180)
    if r.status_code != 200:
        # show detailed server message to debug
        raise RuntimeError(f"Ollama {r.status_code}: {r.text[:500]}")
    data = r.json()
    return data.get("response","").strip()


# ---------- UI ----------
st.set_page_config(page_title="Doc QA (Forecast ⊂ Sales)", page_icon="🧠", layout="wide")
st.title("🧠 QA over Sales + Forecast (forecast subset only)")

with st.sidebar:
    st.markdown("**Settings**")
    top_k = st.slider("Top-K chunks", 1, 10, 5)
    temp  = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    max_t = st.slider("Max answer tokens", 64, 512, 250, 32)
    st.caption(f"Index: `{INDEX_PATH}` | Meta: `{META_PATH}` | Model: `{OLLAMA_MODEL}`")

q = st.text_input("Ask a question (e.g., 'Compare sales vs forecast for HISSPL10HPINV in JAN 2025')", "")
ask = st.button("Ask")

if ask and q.strip():
    with st.spinner("Retrieving…"):
        hits = retrieve(q, k=top_k)

    st.subheader("Retrieved Context")
    st.dataframe(hits[["CIT","score","item_code","site","month_key","text"]], use_container_width=True, height=220)

    prompt = build_prompt(q, hits)
    with st.spinner("Generating answer…"):
        try:
            ans = ask_ollama(prompt, temperature=temp, max_tokens=max_t)
        except Exception as e:
            ans = f"Error contacting Ollama: {e}"

    st.subheader("Answer")
    st.write(ans)

    with st.expander("Show prompt (debug)"):
        st.code(prompt)