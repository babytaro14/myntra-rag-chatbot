"""
RAG core for the Myntra fashion chatbot.

Retrieval modes
---------------
- semantic : OpenAI query embedding -> ChromaDB cosine search (meaning-based)
- hybrid   : semantic  +  BM25 keyword search, fused with Reciprocal Rank Fusion
             (catches exact terms like brand names / fabrics that pure semantic
             search can miss, while keeping meaning-based matching)

Then: relevance gate -> grounded gpt-4o-mini answer (or fallback).

Public API:
    answer(query, hybrid=False, ...) -> dict(answer, sources, used_fallback, retrieved)
    retrieve(query, ...)             -> semantic hits
    retrieve_hybrid(query, ...)      -> fused hits
"""

import re
from functools import lru_cache

# Streamlit Cloud ships an old system sqlite3 that ChromaDB rejects. If the
# pysqlite3 backport is installed, swap it in before importing chromadb.
try:  # pragma: no cover
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except ImportError:
    pass  # local machine has a new-enough sqlite3

import chromadb
import pandas as pd
from openai import OpenAI
from rank_bm25 import BM25Okapi

import config

# If even the closest product is farther than this cosine distance, treat the
# question as out-of-catalogue and fall back instead of forcing an answer.
RELEVANCE_MAX_DISTANCE = 0.72

# How many candidates each retriever contributes before fusion.
CANDIDATE_POOL = 20

SYSTEM_PROMPT = (
    "You are a helpful shopping assistant for a Myntra-style fashion catalogue "
    "(mostly women's apparel). Answer the user's question USING ONLY the product "
    "context provided below. Follow these rules strictly:\n"
    "1. Ground every claim (product name, brand, price, fabric, colour, fit, "
    "occasion, rating) in the given context. Never invent products or details.\n"
    "2. If the context does not contain enough information to answer, say so "
    "clearly (e.g. \"I couldn't find that in the catalogue\") and suggest how the "
    "user might rephrase or broaden their search. Do NOT guess.\n"
    "3. When recommending products, cite the product name, brand and price (in INR).\n"
    "4. Be concise and friendly. Use a short list when suggesting multiple items."
)

FALLBACK_MESSAGE = (
    "I couldn't find anything relevant in the Myntra catalogue for that. "
    "This catalogue is mostly women's fashion (kurtas, dresses, tops, ethnic & "
    "western wear). Try rephrasing, or ask about a product type, fabric, colour, "
    "occasion or price range."
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


# ---------------------------------------------------------------- clients ----
@lru_cache(maxsize=1)
def _openai() -> OpenAI:
    config.require_openai_key()
    return OpenAI(api_key=config.OPENAI_API_KEY)


def _embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=config.EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


@lru_cache(maxsize=1)
def _collection():
    """Return the ChromaDB collection, building it from the cleaned CSV if empty.

    This makes the app self-bootstrapping: on a fresh host (e.g. Streamlit Cloud)
    where only the cleaned CSV is shipped, the vector index is rebuilt on first
    use with the SAME chunking + embeddings as the local build.
    """
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    col = client.get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    if col.count() > 0:
        return col

    # ---- build on startup ----
    oai = _openai()
    df = pd.read_csv(config.PROCESSED_CSV)
    batch = config.EMBED_BATCH
    for start in range(0, len(df), batch):
        chunk = df.iloc[start:start + batch]
        vectors = _embed_batch(oai, chunk["retrieval_text"].astype(str).tolist())
        col.add(
            ids=[str(int(r["product_id"])) for _, r in chunk.iterrows()],
            embeddings=vectors,
            documents=chunk["retrieval_text"].astype(str).tolist(),
            metadatas=[{
                "name": str(r["name"]), "brand": str(r["brand"]),
                "colour": str(r["colour"]),
                "price": float(r["price"]) if pd.notna(r["price"]) else -1.0,
                "avg_rating": float(r["avg_rating"]) if pd.notna(r["avg_rating"]) else -1.0,
            } for _, r in chunk.iterrows()],
        )
    return col


@lru_cache(maxsize=1)
def _img_map() -> dict:
    """Map product_id -> image URL, read from the cleaned CSV (for the UI)."""
    if not config.PROCESSED_CSV.exists():
        return {}
    df = pd.read_csv(config.PROCESSED_CSV, usecols=["product_id", "img"]).dropna(subset=["product_id"])
    return {str(int(pid)): url for pid, url in zip(df["product_id"], df["img"])}


def ensure_index():
    """Trigger the (cached) build-if-empty. Call once at app startup."""
    col = _collection()
    return col.count()


@lru_cache(maxsize=1)
def _bm25_index():
    """Build an in-memory BM25 keyword index over the same corpus Chroma holds.

    Returns (bm25, ids, records) where records[id] = {document, metadata}.
    """
    df = pd.read_csv(config.PROCESSED_CSV)
    ids, corpus_tokens, records = [], [], {}
    for _, row in df.iterrows():
        _id = str(int(row["product_id"]))
        text = str(row["retrieval_text"])
        ids.append(_id)
        corpus_tokens.append(_tokenize(text))
        records[_id] = {
            "document": text,
            "metadata": {
                "name": str(row["name"]),
                "brand": str(row["brand"]),
                "colour": str(row["colour"]),
                "price": float(row["price"]) if pd.notna(row["price"]) else -1.0,
                "avg_rating": float(row["avg_rating"]) if pd.notna(row["avg_rating"]) else -1.0,
            },
        }
    return BM25Okapi(corpus_tokens), ids, records


def _passes_filters(meta: dict, price_max, colour) -> bool:
    if price_max is not None and not (0 <= meta.get("price", -1) <= float(price_max)):
        return False
    if colour and meta.get("colour", "").lower() != colour.lower():
        return False
    return True


# ------------------------------------------------------------- retrievers ----
def embed_query(query: str) -> list[float]:
    resp = _openai().embeddings.create(model=config.EMBED_MODEL, input=[query])
    return resp.data[0].embedding


def retrieve(query: str, k: int = config.TOP_K,
             price_max: float | None = None,
             colour: str | None = None) -> list[dict]:
    """Semantic retrieval: k most similar products (optional metadata filter)."""
    conds = []
    if price_max is not None:
        conds.append({"price": {"$lte": float(price_max)}})
    if colour:
        conds.append({"colour": colour})
    where = conds[0] if len(conds) == 1 else ({"$and": conds} if conds else None)

    res = _collection().query(
        query_embeddings=[embed_query(query)],
        n_results=k,
        where=where,
    )
    hits = []
    for doc, meta, dist, _id in zip(
        res["documents"][0], res["metadatas"][0],
        res["distances"][0], res["ids"][0]
    ):
        hits.append({"id": _id, "document": doc, "metadata": meta,
                     "distance": dist, "source": "semantic"})
    return hits


def _rrf_fuse(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion: merge several ranked id-lists into one ranking."""
    scores: dict[str, float] = {}
    for lst in rank_lists:
        for rank, _id in enumerate(lst):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def retrieve_hybrid(query: str, k: int = config.TOP_K,
                    price_max: float | None = None,
                    colour: str | None = None) -> list[dict]:
    """Hybrid retrieval: fuse semantic + BM25 keyword results with RRF."""
    bm25, ids, records = _bm25_index()

    # 1. semantic candidates (already filtered by Chroma `where`)
    sem_hits = retrieve(query, k=CANDIDATE_POOL, price_max=price_max, colour=colour)
    sem_ids = [h["id"] for h in sem_hits]
    sem_dist = {h["id"]: h["distance"] for h in sem_hits}

    # 2. BM25 candidates, then apply the same metadata filter
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
    bm25_ids = []
    for i in ranked:
        _id = ids[i]
        if scores[i] <= 0:
            break
        if _passes_filters(records[_id]["metadata"], price_max, colour):
            bm25_ids.append(_id)
        if len(bm25_ids) >= CANDIDATE_POOL:
            break

    # 3. fuse and take top-k
    fused = _rrf_fuse([sem_ids, bm25_ids])[:k]

    hits = []
    for _id in fused:
        rec = records.get(_id)
        if rec is None:
            continue
        in_sem = _id in sem_dist
        in_bm25 = _id in bm25_ids
        hits.append({
            "id": _id,
            "document": rec["document"],
            "metadata": rec["metadata"],
            "distance": sem_dist.get(_id),  # None if BM25-only
            "source": "both" if (in_sem and in_bm25) else ("semantic" if in_sem else "keyword"),
        })
    # expose the best semantic distance so the caller can run the relevance gate
    best_dist = min(sem_dist.values()) if sem_dist else None
    return hits, best_dist


# -------------------------------------------------------------- generation ---
def _build_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        price = f"Rs{m['price']:.0f}" if m.get("price", -1) >= 0 else "N/A"
        rating = f"{m['avg_rating']:.1f}/5" if m.get("avg_rating", -1) >= 0 else "unrated"
        blocks.append(
            f"[Product {i}] {m['name']} | Brand: {m['brand']} | Colour: {m['colour']} "
            f"| Price: {price} | Rating: {rating}\n{h['document']}"
        )
    return "\n\n".join(blocks)


def _sources(hits: list[dict]) -> list[dict]:
    imgs = _img_map()
    out = []
    for h in hits:
        m = h["metadata"]
        out.append({
            "id": h["id"],
            "name": m["name"], "brand": m["brand"], "price": m["price"],
            "colour": m["colour"], "avg_rating": m.get("avg_rating", -1),
            "img": imgs.get(h["id"]),
            "distance": round(h["distance"], 3) if h.get("distance") is not None else None,
            "source": h.get("source", "semantic"),
        })
    return out


def answer(query: str, k: int = config.TOP_K, hybrid: bool = False,
           price_max: float | None = None, colour: str | None = None) -> dict:
    """Full RAG answer with grounding + fallback. Set hybrid=True for BM25 fusion."""
    if hybrid:
        hits, best_dist = retrieve_hybrid(query, k=k, price_max=price_max, colour=colour)
    else:
        hits = retrieve(query, k=k, price_max=price_max, colour=colour)
        best_dist = hits[0]["distance"] if hits else None

    # Relevance gate: nothing semantically close -> fallback without the LLM.
    if not hits or best_dist is None or best_dist > RELEVANCE_MAX_DISTANCE:
        return {"answer": FALLBACK_MESSAGE, "sources": [], "used_fallback": True,
                "retrieved": hits if hits else []}

    context = _build_context(hits)
    user_msg = (
        f"Product context:\n{context}\n\n"
        f"User question: {query}\n\n"
        "Answer using only the context above. If it is insufficient, say so."
    )
    resp = _openai().chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    text = resp.choices[0].message.content.strip()
    return {"answer": text, "sources": _sources(hits), "used_fallback": False,
            "retrieved": hits}


if __name__ == "__main__":
    # Compare semantic vs hybrid on a query with an exact brand term.
    for q in ["Do you have anything from Libas in black",
              "What is the capital of France?"]:
        print("=" * 90)
        print("Q:", q)
        for mode in (False, True):
            out = answer(q, hybrid=mode)
            tag = "HYBRID" if mode else "SEMANTIC"
            print(f"\n[{tag}] fallback={out['used_fallback']}")
            print(out["answer"][:300])
            if out["sources"]:
                print("sources:", [(s["name"][:32], s["source"]) for s in out["sources"][:3]])
