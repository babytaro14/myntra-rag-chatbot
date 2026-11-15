"""
Clean and prepare the Myntra Fashion dataset for RAG.

Input : data/raw/Fashion Dataset.csv  (14,330 rows, HTML-laden descriptions)
Output: data/processed/products_clean.csv

Steps
-----
1. Drop fully-null rows and deduplicate on p_id.
2. Strip HTML from the description.
3. Parse the stringified p_attributes dict and pull out useful fashion attributes.
4. Build one composite 'retrieval_text' per product (what we embed).
5. Keep brand / colour / price / avg_rating as filterable metadata.
"""

import ast
import html
import re
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/Fashion Dataset.csv")
OUT = Path("data/processed/products_clean.csv")

# p_attributes keys worth surfacing for retrieval (skip noisy/ID-like ones)
ATTR_KEYS = [
    "Fabric", "Top Fabric", "Bottom Fabric", "Dupatta Fabric",
    "Occasion", "Main Trend", "Print or Pattern Type", "Pattern",
    "Neck", "Sleeve Length", "Sleeve Styling",
    "Length", "Bottom Type", "Dupatta", "Shape", "Fit", "Waist Rise",
    "Weave Type", "Ornamentation", "Hemline", "Closure",
]

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_html(text: str) -> str:
    """Strip HTML tags/entities and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.replace("<br>", " ").replace("<li>", " • ")
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = text.replace("•", "-")        # bullet -> dash for plain text
    text = _WS_RE.sub(" ", text).strip()
    return text


def parse_attrs(raw) -> dict:
    """p_attributes is a stringified python dict; parse safely, drop 'NA'."""
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        d = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(d, dict):
        return {}
    return {k: v for k, v in d.items()
            if isinstance(v, str) and v.strip() and v.strip().upper() != "NA"}


def attrs_to_text(attrs: dict) -> str:
    parts = []
    for key in ATTR_KEYS:
        if key in attrs:
            parts.append(f"{key}: {attrs[key]}")
    return ". ".join(parts)


def build_retrieval_text(row) -> str:
    price = row["price"]
    price_str = f"₹{int(price)}" if pd.notna(price) else "price N/A"
    rating = row["avg_rating"]
    rating_str = f"{float(rating):.1f}/5" if pd.notna(rating) else "unrated"

    header = (
        f"{row['name']} by {row['brand']}. "
        f"Colour: {row['colour']}. Price: {price_str}. Rating: {rating_str}."
    )
    desc = row["description_clean"]
    attr_text = row["attr_text"]

    segments = [header]
    if desc:
        segments.append(desc)
    if attr_text:
        segments.append(attr_text)
    return " ".join(segments)


def main() -> None:
    df = pd.read_csv(RAW)
    print(f"raw rows: {len(df)}")

    # 1. drop rows with no name/description, dedup on product id
    df = df.dropna(subset=["name", "description"])
    before = len(df)
    df = df.drop_duplicates(subset=["p_id"], keep="first")
    print(f"after dropna + dedup: {len(df)} (removed {before - len(df)} dup p_id)")

    # 2. clean description
    df["description_clean"] = df["description"].map(clean_html)

    # 3. parse attributes
    attrs = df["p_attributes"].map(parse_attrs)
    df["attr_text"] = attrs.map(attrs_to_text)

    # 4. composite retrieval text
    df["retrieval_text"] = df.apply(build_retrieval_text, axis=1)

    # 5. tidy final columns
    out = pd.DataFrame({
        "product_id": df["p_id"].astype("Int64"),
        "name": df["name"],
        "brand": df["brand"],
        "colour": df["colour"],
        "price": df["price"],
        "avg_rating": df["avg_rating"],
        "rating_count": df["ratingCount"],
        "img": df["img"],
        "description": df["description_clean"],
        "attributes": df["attr_text"],
        "retrieval_text": df["retrieval_text"],
    })

    # drop any rows whose retrieval text is too thin to be useful
    out = out[out["retrieval_text"].str.len() >= 40].reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    # ---- report ----
    rl = out["retrieval_text"].str.len()
    print(f"\nsaved {len(out)} products -> {OUT}")
    print(f"retrieval_text length: min {rl.min()}, median {int(rl.median())}, "
          f"mean {rl.mean():.0f}, max {rl.max()}")
    print(f"rows with rating: {out['avg_rating'].notna().sum()} "
          f"({out['avg_rating'].notna().mean()*100:.0f}%)")
    print(f"unique brands: {out['brand'].nunique()}")
    print("\n===== 2 sample retrieval_text =====")
    for t in out["retrieval_text"].head(2):
        print("-" * 80)
        print(t[:650])


if __name__ == "__main__":
    main()
