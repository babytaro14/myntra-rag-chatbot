"""
Build the vector store: embed every product's retrieval_text with OpenAI and
store vectors + metadata in a persistent ChromaDB collection.

Run:  python src/build_index.py
Re-run with --limit N to index only the first N products (quick test).
"""

import argparse

import chromadb
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

import config


def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=config.EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Only index the first N rows (for a quick test).")
    ap.add_argument("--reset", action="store_true",
                    help="Delete and rebuild the collection.")
    args = ap.parse_args()

    config.require_openai_key()
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    df = pd.read_csv(config.PROCESSED_CSV)
    if args.limit:
        df = df.head(args.limit)
    print(f"Indexing {len(df)} products with {config.EMBED_MODEL}")

    chroma = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if args.reset:
        try:
            chroma.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass
    # cosine space matches OpenAI embeddings well
    collection = chroma.get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    batch = config.EMBED_BATCH
    for start in tqdm(range(0, len(df), batch), unit="batch"):
        chunk = df.iloc[start:start + batch]
        texts = chunk["retrieval_text"].astype(str).tolist()
        vectors = embed_batch(client, texts)

        ids, metadatas = [], []
        for _, row in chunk.iterrows():
            ids.append(str(int(row["product_id"])))
            metadatas.append({
                "name": str(row["name"]),
                "brand": str(row["brand"]),
                "colour": str(row["colour"]),
                "price": float(row["price"]) if pd.notna(row["price"]) else -1.0,
                "avg_rating": float(row["avg_rating"]) if pd.notna(row["avg_rating"]) else -1.0,
            })

        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,          # stored so retrieval can return the context
            metadatas=metadatas,
        )

    print(f"\nDone. Collection '{config.COLLECTION_NAME}' now holds "
          f"{collection.count()} vectors at {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
