# 🛍️ Myntra Fashion RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about a
Myntra-style fashion catalogue — product discovery, fabric, fit, price, colour,
occasion — **grounded in retrieved product data**, with a Streamlit chat UI.

Built for the RAG Chatbot assignment. Baseline RAG **+ Hybrid RAG** (semantic +
BM25 keyword) enhancement.

![pipeline](https://img.shields.io/badge/pipeline-data→clean→embed→ChromaDB→retrieve→LLM-blue)

## What it does
- Answers fashion queries using **only** the catalogue (14,223 products), not the
  LLM's general knowledge.
- **Hybrid retrieval**: semantic vector search (OpenAI + ChromaDB) fused with BM25
  keyword search via Reciprocal Rank Fusion.
- **Grounded answers** citing product name, brand and price.
- **Fallback** for unsupported / off-topic questions (won't hallucinate).
- Sidebar controls: hybrid toggle, top-k, price & colour filters, source badges.

## Tech stack
| Layer | Choice |
|-------|--------|
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| Vector store | ChromaDB (persistent, cosine) |
| Keyword search | `rank-bm25` (BM25Okapi) |
| UI | Streamlit |
| Data | [Myntra Fashion Dataset](https://www.kaggle.com/datasets/hiteshsuthar101/myntra-fashion-product-dataset) (Kaggle, PDDL) |

## Project structure
```
myntra-rag-chatbot/
├── app.py                 # Streamlit chat UI
├── requirements.txt
├── kaggle-rag-chatbot.env # secrets (KAGGLE_KEY, OPENAI_API_KEY) — git-ignored
├── src/
│   ├── config.py          # paths, models, env loading
│   ├── prepare_data.py     # clean raw CSV -> products_clean.csv
│   ├── build_index.py      # embed products -> ChromaDB
│   └── rag.py              # retrieval (semantic + hybrid) + grounded answer
├── tests/
│   └── run_tests.py        # runs the evaluation questions
├── docs/
│   ├── PIPELINE.md         # data -> answer explanation
│   ├── RAG_TECHNIQUES.md   # baseline + hybrid RAG
│   └── TESTING.md          # test cases + failure analysis
├── data/
│   ├── raw/                # Fashion Dataset.csv (downloaded)
│   └── processed/          # products_clean.csv (generated)
└── chroma_db/              # vector store (generated)
```

## Setup

### 1. Install
```bash
cd myntra-rag-chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add credentials
Create / edit `kaggle-rag-chatbot.env` (git-ignored):
```
KAGGLE_KEY=KGAT_xxxxxxxx          # newer Kaggle API token (for downloading data)
OPENAI_API_KEY=sk-xxxxxxxx        # required for embeddings + answers
```
> The Kaggle key is only needed to re-download the dataset. The OpenAI key is
> required to build the index and run the chatbot.

### 3. Get the data (if `data/raw/` is empty)
```bash
set -a && . ./kaggle-rag-chatbot.env && set +a
kaggle datasets download -d hiteshsuthar101/myntra-fashion-product-dataset \
  -f "Fashion Dataset.csv" -p data/raw
cd data/raw && unzip -o "Fashion%20Dataset.csv.zip" && rm "Fashion%20Dataset.csv.zip" && cd ../..
```

### 4. Build the pipeline
```bash
python src/prepare_data.py          # clean -> data/processed/products_clean.csv
python src/build_index.py --reset    # embed all products -> chroma_db/  (~2 min, ~$0.04)
```
Quick test build: `python src/build_index.py --reset --limit 50`

### 5. Run the chatbot
```bash
streamlit run app.py
```
Open http://localhost:8501.

## Testing
```bash
python tests/run_tests.py
```
Runs the factual / comparison / multi-part / out-of-dataset questions and prints
answers + retrieved products. See [docs/TESTING.md](docs/TESTING.md) for the
analysis and documented failure cases.

## Demo tip (5-minute presentation)
Toggle **Hybrid search OFF**, ask a brand/fabric-specific question (e.g. *"Do you
have a Biba anarkali kurta?"*), then toggle it **ON** and ask again — the "both"
badges and tighter brand-locked results show the enhancement working live.

## Notes & limitations
- The catalogue is **women's fashion** (kurtas, dresses, tops, ethnic/western).
- Natural-language price/colour constraints aren't auto-parsed yet — use the
  sidebar filters. See failure analysis in [docs/TESTING.md](docs/TESTING.md).
