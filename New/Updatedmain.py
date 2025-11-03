# ====== Initialization Cell (Run ONCE) ======
# Loads embedding model, FAISS index, metadata, BM25, (optional) reranker.
# After this cell, call:   answer_query("your question")   as many times as you want.

import os
import json
import faiss
import numpy as np
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler
# -------- CONFIG (update paths if needed) ----------
API_MAPPING_FILE = r"C:\Users\Arpit.x.Tripathi\Downloads\Rag_chatbot\api_mapping.json"    # output from Step 2
FAISS_INDEX_FILE = r"C:\Users\Arpit.x.Tripathi\Downloads\Rag_chatbot\api_index.faiss"     # output from Step 2
EMBED_MODEL_NAME = r"C:\Users\Arpit.x.Tripathi\Downloads\Rag_chatbot\models\all-MiniLM-L6-v2"

# Optional local reranker (cross-encoder). If not found, code will fall back to RRF (non-LLM).
RERANKER_MODEL = r"C:\Users\Arpit.x.Tripathi\Downloads\Rag_chatbot\Reranking-Model\cross-encoder-ms-marco-MiniLM-L-6-v2"

# --------- LOAD / INIT ----------
print("Loading API metadata...")
with open(API_MAPPING_FILE, "r", encoding="utf-8") as f:
    api_metadata = json.load(f)
print(f"Loaded {len(api_metadata)} API entries")

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_FILE)
print("FAISS index loaded. vectors:", index.ntotal)

print(f"Loading embedding model: {EMBED_MODEL_NAME} ...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
print("Embedding model loaded.")

# ---------- Text cleaning / tokenization ----------
def clean_text_tokens(text: str):
    if not isinstance(text, str):
        text = ""
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text).lower()
    tokens = text.split()
    # Optional stopwords removal could be added here
    return tokens

# Add parameters & tags into corpus text to improve relevance
def metadata_to_text(m):
    params = m.get('parameters', []) or []
    ptext = " ".join([f"{p.get('in','')}:{p.get('name','')}:{p.get('type','')}" for p in params])
    rtext = " ".join([f"{code}:{resp.get('description','')}" for code, resp in (m.get('responses') or {}).items()])
    return " | ".join([
        str(m.get('path','')),
        str(m.get('method','')),
        str(m.get('summary','')),
        str(m.get('description','')),
        "tags:" + " ".join(m.get('tags',[])),
        "params:" + ptext,
        "responses:" + rtext
    ])

print("Building BM25 index...")
bm25_corpus = []
for m in api_metadata:
    txt = metadata_to_text(m)
    bm25_corpus.append(clean_text_tokens(txt))
bm25 = BM25Okapi(bm25_corpus)
print("BM25 ready.")

# --------- (Optional) Reranker loading ----------
reranker = None
reranker_device = "cpu"
try:
    if os.path.exists(RERANKER_MODEL):
        from sentence_transformers import CrossEncoder  # from sentence-transformers
        print(f"Loading reranker from local path: {RERANKER_MODEL} ...")
        reranker = CrossEncoder(RERANKER_MODEL, device=reranker_device)
        print("Reranker loaded.")
    else:
        print(f"Reranker path not found: {RERANKER_MODEL}. Will use RRF fallback (no reranker).")
except Exception as e:
    print("Failed to load reranker; continuing without it. Error:", str(e))
    reranker = None

# --------- Utility functions ----------
def embed_query(query: str):
    vec = embed_model.encode([query], convert_to_numpy=True)
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    return vec.astype("float32")

def dense_search(query_vec, top_k=50):
    D, I = index.search(query_vec, top_k)
    results = []
    for idx, dist in zip(I[0], D[0]):
        if idx < 0:
            continue
        # Convert L2 distance to a similarity-like score (bounded, higher=better)
        sim = 1.0 / (1.0 + float(dist))
        results.append((int(idx), sim))
    return results

def bm25_search(query, top_k=50):
    tokens = clean_text_tokens(query)
    scores = bm25.get_scores(tokens)
    ranked = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in ranked]

def _minmax(arr):
    """Safe MinMax scaling for list/np array; returns zeros if degenerate."""
    arr = np.array(arr, dtype=float).reshape(-1,1)
    if arr.size == 0:
        return np.array([])
    if np.allclose(arr.min(), arr.max()):
        return np.zeros_like(arr).flatten()
    return MinMaxScaler().fit_transform(arr).flatten()

def hybrid_retrieve(query, top_k=20, bm25_weight=0.35):
    """
    Hybrid retrieval combining dense (FAISS) and BM25 with MinMax normalization.
    Returns list of dicts: {'id', 'score', 'meta'}
    """
    qvec = embed_query(query)
    dense = dense_search(qvec, top_k*2)
    bm25_res = bm25_search(query, top_k*2)

    d_norm = _minmax([s for _, s in dense]) if dense else np.array([])
    b_norm = _minmax([s for _, s in bm25_res]) if bm25_res else np.array([])

    dense_dict = {dense[i][0]: float(d_norm[i]) for i in range(len(dense))} if dense.size != 0 else {}
    bm25_dict = {bm25_res[i][0]: float(b_norm[i]) for i in range(len(bm25_res))} if bm25_res.size != 0 else {}

    all_ids = set(list(dense_dict.keys()) + list(bm25_dict.keys()))
    hybrid = []
    for idx in all_ids:
        d = dense_dict.get(idx, 0.0)
        b = bm25_dict.get(idx, 0.0)
        score = (1 - bm25_weight) * d + bm25_weight * b
        hybrid.append((idx, float(score)))
    hybrid = sorted(hybrid, key=lambda x: x[1], reverse=True)[:top_k]
    return [{"id": idx, "score": s, "meta": api_metadata[idx]} for idx, s in hybrid]

def rerank_candidates(query: str, candidates: list, top_k=5):
    """
    Use cross-encoder to rerank candidate list if available.
    Returns top_k list with 'rerank_score'.
    """
    if not candidates:
        return []
    if reranker is None:
        # No reranker available, return top_k by hybrid score
        ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]
        for r in ranked:
            r["rerank_score"] = None
        return ranked

    # Compose pair texts: (query, metadata_text)
    pair_texts = []
    for c in candidates:
        m = c["meta"]
        text = metadata_to_text(m)
        pair_texts.append((query, text))

    scores = reranker.predict(pair_texts)
    for i, s in enumerate(scores):
        candidates[i]["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
    return ranked
# ---------- Helper: build request examples ----------
def _extract_params(m):
    params = m.get("parameters", []) or []
    # normalize minimal fields
    out = []
    for p in params:
        out.append({
            "name": p.get("name", ""),
            "in": p.get("in", "query"),
            "type": (p.get("type") or (p.get("schema", {}) or {}).get("type") or "string"),
            "required": bool(p.get("required", False)),
            "description": p.get("description", "")
        })
    return out

def _build_examples(m, base_url=None):
    """
    Generate cURL and Python requests examples with placeholders.
    """
    path = m.get("path","")
    method = (m.get("method","GET") or "GET").upper()
    params = _extract_params(m)

    # Replace path template {id} -> <id>
    curl_path = path
    for p in params:
        if p["in"] == "path" and p["name"]:
            curl_path = re.sub(r"\{" + re.escape(p["name"]) + r"\}", f"<{p['name']}>", curl_path)

    # Build query string
    query_params = [p for p in params if p["in"] == "query"]
    qs_parts = []
    for p in query_params:
        placeholder = f"<{p['name']}>"
        qs_parts.append(f"{p['name']}={placeholder}")
    qs = "&".join(qs_parts)
    url = (base_url or "https://api.example.com") + curl_path + (f"?{qs}" if qs else "")

    # Body params
    body_params = [p for p in params if p["in"] == "body" and p["name"] not in (None, "", "(body)")]
    body_obj = {p["name"]: f"<{p['name']}>" for p in body_params}
    body_json = json.dumps(body_obj, indent=2) if body_obj else "{}"

    headers = [
        '-H "Content-Type: application/json"',
        '-H "Authorization: Bearer <token>"'
    ]

    curl = f"curl -X {method} \\\n  \"{url}\" \\\n  " + " \\\n  ".join(headers)
    if method in ["POST", "PUT", "PATCH"] and body_obj:
        curl += f" \\\n  -d '{body_json}'"

    py = f'''import requests, json

url = "{url}"
headers = {{
    "Content-Type": "application/json",
    "Authorization": "Bearer <token>"
}}
payload = {body_json}

resp = requests.{method.lower()}(url, headers=headers, json=(payload if payload != {{}} else None), timeout=30)
print("Status:", resp.status_code)
try:
    print(json.dumps(resp.json(), indent=2))
except Exception:
    print(resp.text)
'''
    return curl, py

# ---------- Deterministic (non-LLM) plan ----------
def template_plan(query: str, reranked: list, base_url=None):
    lines = []
    lines.append(f"User Query: {query}")
    lines.append(f"Generated at: {datetime.utcnow().isoformat()}Z")
    lines.append("\nSuggested API Workflow (top results):\n")
    for i, c in enumerate(reranked, start=1):
        m = c["meta"]
        lines.append(f"Step {i}: {m.get('summary') or m.get('path')}")
        lines.append(f"  Endpoint: {m.get('path')}  Method: {m.get('method')}")
        params = _extract_params(m)
        if params:
            # show parameter names and types
            param_lines = []
            for p in params:
                pname = p.get("name", "<name>")
                ptype = p.get("type") or "string"
                required = p.get("required", False)
                where = p.get("in","query")
                star = "*" if required else ""
                param_lines.append(f"{where}:{pname}({ptype}){star}")
            lines.append("  Parameters: " + ", ".join(param_lines))

        # Build examples
        curl, py = _build_examples(m, base_url=base_url)
        lines.append("  Example (cURL):")
        lines.append("    " + "\n    ".join(curl.splitlines()))
        lines.append("  Example (Python requests):")
        lines.append("    " + "\n    ".join(py.splitlines()))
        lines.append("")  # blank line
    return "\n".join(lines)

# ---------- Reciprocal Rank Fusion (fallback if no reranker) ----------
def rrf_fusion(dense_results, bm25_results, k=60, top_k=5):
    """
    dense_results: list[(idx, score)] sorted desc by dense
    bm25_results:  list[(idx, score)] sorted desc by bm25
    Returns list of dicts like hybrid_retrieve but using RRF.
    """
    def rank_map(res):
        # res is list[(idx, score)] sorted desc by score
        return {i: r for r, (i, _) in enumerate(res, start=1)}

    d_sorted = sorted(dense_results, key=lambda x: x[1], reverse=True)
    b_sorted = sorted(bm25_results, key=lambda x: x[1], reverse=True)
    r_dense = rank_map(d_sorted)
    r_bm25 = rank_map(b_sorted)

    all_ids = set(r_dense.keys()) | set(r_bm25.keys())
    fused = []
    for i in all_ids:
        s = (1 / (k + r_dense.get(i, 9999))) + (1 / (k + r_bm25.get(i, 9999)))
        fused.append((i, s))
    fused = sorted(fused, key=lambda x: x[1], reverse=True)[:top_k]
    return [{"id": i, "score": s, "meta": api_metadata[i]} for i, s in fused]

# ---------- Public function: Answer a query ----------
def answer_query(query: str, top_k=5, bm25_weight=0.35, use_reranker=True, base_url=None, verbose=True):
    """
    End-to-end:
      - Hybrid retrieve (FAISS + BM25)
      - Optional rerank (cross-encoder if available) else RRF fallback
      - Produce deterministic plan with sample calls

    Returns a dict:
      {
        "query": ...,
        "candidates": [ {path, method, score, rerank_score}, ... ],
        "plan_text": "...",
        "raw": { "hybrid": [...], "reranked": [...] }
      }
    """
    # Step 1: Hybrid retrieve
    hybrid = hybrid_retrieve(query, top_k=max(top_k*2, 10), bm25_weight=bm25_weight)

    # Step 2: Rerank
    if use_reranker and reranker is not None:
        reranked = rerank_candidates(query, hybrid, top_k=top_k)
    else:
        # RRF fallback (non-LLM; robust)
        qvec = embed_query(query)
        dense = dense_search(qvec, top_k=max(top_k*2, 10))
        bm25_res = bm25_search(query, top_k=max(top_k*2, 10))
        reranked = rrf_fusion(dense, bm25_res, top_k=top_k)

        # For consistent structure
        for r in reranked:
            r["rerank_score"] = None

    # Step 3: Build plan (deterministic, includes examples)
    plan_text = template_plan(query, reranked, base_url=base_url)

    # Pretty console output (optional)
    if verbose:
        print("\nTop candidates:")
        for i, r in enumerate(reranked, 1):
            m = r["meta"]
            s = r.get("rerank_score", None)
            print(f"{i}. {m.get('path')}  [{m.get('method')}]  "
                  f"hybrid_score={r.get('score'):.3f}"
                  + (f"  rerank_score={s:.3f}" if s is not None else ""))

        print("\n" + "="*80)
        print(plan_text)
        print("="*80)

    # Compact view for caller
    candidates_view = [{
        "path": r["meta"].get("path"),
        "method": r["meta"].get("method"),
        "score": r.get("score"),
        "rerank_score": r.get("rerank_score")
    } for r in reranked]

    return {
        "query": query,
        "candidates": candidates_view,
        "plan_text": plan_text,
        "raw": {
            "hybrid": hybrid,
            "reranked": reranked
        }
    }

print("\nInitialization complete ✅  You can now call:")
print(' answer_query("Which API is responsible for getting the booking details?")')