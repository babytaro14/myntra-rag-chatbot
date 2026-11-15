"""
Streamlit chat UI for the Myntra Fashion RAG chatbot.

Run:  streamlit run app.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

# set_page_config MUST be the first Streamlit command on the page.
st.set_page_config(page_title="Myntra Fashion Assistant", page_icon="🛍️", layout="wide")

# On Streamlit Cloud the API key is provided via st.secrets; bridge it into the
# environment so src/config.py picks it up the same way as the local .env file.
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass  # malformed/absent secrets file -> fall back to .env locally

import config
import rag


@st.cache_resource(show_spinner="Building the product index (first load only, ~2 min)...")
def _warm_index() -> int:
    """Build the vector index on first run if it isn't present yet."""
    return rag.ensure_index()


_warm_index()

# ---------------- Styling ----------------
st.markdown(
    """
    <style>
      .stApp { background: #fafafa; }
      h1 { color: #ff3f6c; }  /* Myntra pink */
      .product-card {
        border: 1px solid #ececec; border-radius: 12px; padding: 10px;
        background: #ffffff; height: 100%;
      }
      .product-card .pname { font-weight: 600; font-size: 0.82rem; line-height: 1.2;
        margin: 6px 0 2px; color: #282c3f; }
      .product-card .pmeta { font-size: 0.74rem; color: #7e818c; }
      .product-card .pprice { font-weight: 700; color: #282c3f; font-size: 0.9rem; }
      .badge { font-size: 0.65rem; padding: 1px 6px; border-radius: 8px;
        background: #fff0f3; color: #ff3f6c; border: 1px solid #ffd4de; }
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = [
    "Suggest a black cotton kurta for a festive occasion",
    "Do you have a Biba anarkali kurta?",
    "Compare a Biba anarkali with a Libas kurta set",
    "Show me a red dupatta with gotta patti work",
]

BADGE = {"both": "🔵🟢 both", "semantic": "🔵 semantic", "keyword": "🟢 keyword"}


def render_products(sources: list[dict]) -> None:
    """Lay out retrieved products as image cards."""
    if not sources:
        return
    st.caption("🛍️ Products referenced")
    cols = st.columns(min(len(sources), 5))
    for col, s in zip(cols, sources[:5]):
        with col:
            price = f"₹{s['price']:.0f}" if s["price"] >= 0 else "—"
            rating = f"⭐ {s['avg_rating']:.1f}" if s.get("avg_rating", -1) >= 0 else ""
            badge = BADGE.get(s.get("source", "semantic"), "")
            if s.get("img"):
                st.image(s["img"], use_container_width=True)
            st.markdown(
                f"<div class='product-card'>"
                f"<div class='pname'>{s['name'][:55]}</div>"
                f"<div class='pmeta'>{s['brand']} · {s['colour']}</div>"
                f"<div class='pprice'>{price} <span class='pmeta'>{rating}</span></div>"
                f"<span class='badge'>{badge}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("🛍️ Myntra Assistant")
    st.caption("RAG chatbot grounded in a Myntra fashion catalogue.")

    st.subheader("Search settings")
    use_hybrid = st.toggle("Hybrid search (semantic + keyword)", value=True,
                           help="Combine meaning-based vector search with BM25 "
                                "keyword matching. Better at exact terms like "
                                "brand names and fabrics.")
    top_k = st.slider("Products to retrieve (top-k)", 3, 10, config.TOP_K)

    use_price = st.checkbox("Filter by max price")
    price_max = st.slider("Max price (₹)", 200, 20000, 3000, step=100) if use_price else None

    use_colour = st.checkbox("Filter by colour")
    colour = (
        st.selectbox("Colour", ["Black", "White", "Blue", "Red", "Green", "Pink",
                                 "Yellow", "Navy Blue", "Maroon", "Grey", "Beige",
                                 "Purple", "Orange", "Brown"])
        if use_colour else None
    )

    show_sources = st.checkbox("Show product images", value=True)

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Pipeline: data → clean → embeddings (OpenAI) → ChromaDB → "
               "retrieval → grounded LLM answer. Off-topic questions get a fallback.")

# ---------------- Header ----------------
st.title("🛍️ Myntra Fashion Assistant")
st.write("Ask about products, fabrics, fit, price, colour or occasion — "
         "answers come **only** from the catalogue.")

# Example question buttons
st.caption("Try an example:")
ex_cols = st.columns(len(EXAMPLES))
example_clicked = None
for col, ex in zip(ex_cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        example_clicked = ex

# ---------------- Chat state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_products(msg["sources"])


def handle_query(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching the catalogue..."):
            try:
                result = rag.answer(prompt, k=top_k, hybrid=use_hybrid,
                                    price_max=price_max, colour=colour)
            except SystemExit as e:
                result = {"answer": f"⚠️ {e}", "sources": [], "used_fallback": True}
            except Exception as e:
                result = {"answer": f"⚠️ Something went wrong: {e}",
                          "sources": [], "used_fallback": True}
        st.markdown(result["answer"])
        sources = result.get("sources") or []
        if show_sources:
            render_products(sources)
    st.session_state.messages.append({
        "role": "assistant", "content": result["answer"],
        "sources": sources if show_sources else [],
    })


# ---------------- Input ----------------
typed = st.chat_input("Ask about Myntra fashion products...")
query = typed or example_clicked
if query:
    handle_query(query)
