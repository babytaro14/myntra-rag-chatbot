# RAG Techniques Used

## Baseline RAG (mandatory) ✅
A complete working baseline pipeline:
embed query → semantic (cosine) retrieval from ChromaDB → grounded `gpt-4o-mini`
answer with an "insufficient context" fallback. See [PIPELINE.md](PIPELINE.md).

## Enhancement: Hybrid RAG (optional) ✅
We combine **semantic (vector) retrieval** with **keyword retrieval (BM25)** and
merge the results — the first optional enhancement listed in the assignment.

**How it works (`src/rag.py`, `retrieve_hybrid`):**
1. **Semantic search** — OpenAI query embedding → ChromaDB cosine search returns
   up to 20 candidates (matches *meaning*).
2. **Keyword search** — `rank_bm25.BM25Okapi` scores the same 14,223-product
   corpus by exact term overlap and returns up to 20 candidates (matches *exact
   words* like brand names and fabrics).
3. **Fusion** — the two ranked lists are merged with **Reciprocal Rank Fusion**:
   `score(doc) = Σ 1 / (k + rank)` over both lists (k = 60). RRF needs no score
   normalisation, so it robustly combines ChromaDB distances and BM25 scores,
   which live on completely different scales.
4. The top-k fused products go to the LLM.

**Why it helps:** pure semantic search can under-rank exact terms. Example — the
query *"Do you have a Biba anarkali kurta?"* returns **only genuine Biba products**
in hybrid mode, each tagged **both** (semantic *and* keyword agreed), because BM25
locks onto the literal token "Biba" while the vector search keeps the results
meaning-relevant.

**Metadata filtering** (price/colour) is a complementary mechanism also wired in
(`retrieve(..., price_max=, colour=)`), exposed via the Streamlit sidebar. It
enforces *hard* constraints, which neither semantic nor keyword ranking does on
its own.

## Not yet implemented (future work)
- **Query decomposition** — splitting multi-part questions ("red kurta under 3000
  *and* a matching dupatta") into sub-queries, retrieving for each, then combining.
  Would fix Failure case 2 in [TESTING.md](TESTING.md).
- **Query understanding** — parsing price/colour/occasion out of free text into
  the existing metadata filters.
