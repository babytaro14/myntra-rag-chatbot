# Pipeline Explanation

A Retrieval-Augmented Generation (RAG) chatbot for a Myntra-style fashion
catalogue. Every answer is **grounded in retrieved product data**, not the LLM's
general knowledge.

```
 Raw CSV  ->  Clean  ->  Composite text  ->  Embeddings  ->  Vector store
 (Kaggle)    (prep)     (per product)       (OpenAI)        (ChromaDB)
                                                                 |
                                                                 v
 Answer  <-  LLM (grounded + fallback)  <-  Retrieval (semantic + BM25)  <- user query
```

## 1. Data → the dataset
- **Source:** [Myntra Fashion Product Dataset](https://www.kaggle.com/datasets/hiteshsuthar101/myntra-fashion-product-dataset)
  (Kaggle, PDDL / public domain). Scraped from myntra.com — mostly women's apparel.
- **Size:** 14,330 raw rows, 11 columns. After cleaning: **14,223 products**.
- **Key fields:** `name, brand, colour, price, avg_rating, ratingCount,
  description` (rich natural-language text) and `p_attributes` (a structured dict
  of fashion attributes — fabric, neck, sleeve, occasion, pattern, etc.).

## 2. Cleaning (`src/prepare_data.py`)
- Dropped 18 fully-null rows and 89 duplicate `p_id`s.
- **Stripped HTML** from descriptions (the raw text is full of `<br>`, `<ul><li>`
  tags) and collapsed whitespace.
- **Parsed `p_attributes`** (a stringified Python dict) with `ast.literal_eval`,
  discarding `"NA"` values, and pulled out useful attributes (fabric, occasion,
  neck, sleeve, pattern, dupatta, fit…).
- Built one **composite `retrieval_text`** per product — this is what gets
  embedded:

  > *"Khushal K Women Black Ethnic Motifs Printed Kurta with Palazzos by Khushal K.
  > Colour: Black. Price: ₹5099. Rating: 4.4/5. [clean description] Top Fabric:
  > Viscose Rayon. Occasion: Festive. Neck: Mandarin Collar…"*

- Kept `brand, colour, price, avg_rating, rating_count` as **metadata** for
  filtering and for citing sources.
- Output: `data/processed/products_clean.csv`.

## 3. Chunking strategy — **one chunk per product**
Each product's `retrieval_text` is short (median **506 characters**, max ~2,000;
well under the embedding model's limit). So we deliberately **do not split**
products into sub-chunks.

**Why:** a product is the natural unit of retrieval. A user asks "find me a black
cotton kurta," not "find me the sleeve clause of a kurta." Splitting would scatter
one product's price, fabric and occasion across multiple chunks and *hurt*
retrieval. Keeping each product whole means every retrieved chunk is a complete,
self-contained, citable answer unit. (No overlap is needed because there is no
splitting.)

## 4. Embeddings (`src/build_index.py`)
- Model: **OpenAI `text-embedding-3-small`** (1536-dim) — strong quality, very
  cheap (~$0.04 to embed all 14,223 products).
- Sent in batches of 500 texts per request (~29 requests, ~2 minutes total).

## 5. Vector store — **ChromaDB**
- Persistent client at `chroma_db/`, collection `myntra_products`.
- **Cosine** distance (`hnsw:space: cosine`) matches OpenAI embeddings well.
- Each record stores the vector, the full `retrieval_text` (document), and
  metadata (`name, brand, colour, price, avg_rating`).

## 6. Retrieval (`src/rag.py`)
Two modes:

- **Semantic** — embed the query, cosine-search ChromaDB for the top-k nearest
  products. Matches *meaning* ("flowy summer party wear" → georgette festive
  kurtas) even without shared keywords.
- **Hybrid (enhancement)** — run semantic search **and** a **BM25 keyword** search
  over the same corpus, then merge the two ranked lists with **Reciprocal Rank
  Fusion (RRF)**. BM25 nails exact terms (brand names like *Biba*, fabrics like
  *Chanderi*) that pure semantic search can under-rank; semantic keeps the
  meaning-based matching. See [RAG_TECHNIQUES.md](RAG_TECHNIQUES.md).

Both modes support optional **metadata filters** (`price_max`, `colour`) applied
at query time (Chroma `where` for semantic, a post-filter for BM25).

## 7. Context + LLM (grounded generation)
- A **relevance gate** runs first: if the closest product's cosine distance is
  above `0.72`, the query is treated as out-of-catalogue and a **fallback**
  message is returned **without calling the LLM** (saves cost, guarantees no
  hallucination on off-topic questions like "capital of France").
- Otherwise the retrieved products are formatted into a numbered **context block**
  and sent to **`gpt-4o-mini`** with a strict system prompt:
  *answer only from the context, cite name/brand/price, and say so when the
  context is insufficient — never invent products or details.*

## 8. Answer
The model returns a grounded, concise answer. The UI (`app.py`, Streamlit) also
shows the **retrieved products** with their similarity distance and a badge for
how each was found (🔵 semantic · 🟢 keyword · 🔵🟢 both), so grounding is visible.

## Fallback behaviour (unsupported questions)
Two layers guarantee the bot doesn't invent answers:
1. **Pre-LLM relevance gate** — nothing close enough → fixed fallback message.
2. **Prompt-level grounding** — even when products are retrieved, the LLM is
   instructed to decline if they don't actually answer the question (e.g. a
   question about men's shirts returns women's formalwear → the bot says it
   couldn't find it rather than pretending).
