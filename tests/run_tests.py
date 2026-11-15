"""Run representative test questions through the RAG pipeline and print results.

Used to generate the evidence in docs/TESTING.md. Run:
    python tests/run_tests.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import rag  # noqa: E402

TESTS = [
    ("Factual lookup",        "Show me black cotton kurtas for a festive occasion"),
    ("Factual (attribute)",   "What fabric is the InWeave orange kurta with palazzos made of?"),
    ("Comparison",            "Compare a Biba anarkali kurta with a Libas kurta set"),
    ("Multi-part + price",    "I need a red festive kurta under 3000 and a matching dupatta"),
    ("Out-of-dataset (topic)","What is the capital of France?"),
    ("Out-of-scope (men's)",  "Do you have a men's formal shirt?"),
    ("Rating-based",          "What is the best rated anarkali kurta you have?"),
]

for label, q in TESTS:
    out = rag.answer(q, hybrid=True)
    print("=" * 100)
    print(f"[{label}]  Q: {q}")
    print(f"fallback={out['used_fallback']}")
    print("A:", out["answer"])
    if out["sources"]:
        print("retrieved:")
        for s in out["sources"]:
            price = f"Rs{s['price']:.0f}" if s["price"] >= 0 else "N/A"
            print(f"   - {s['name'][:55]} | {s['brand']} | {s['colour']} | {price} "
                  f"| {s['source']} | dist={s['distance']}")
