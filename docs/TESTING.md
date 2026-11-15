# Testing & Evaluation

All results below are **real outputs** from the pipeline (hybrid mode), captured
with `tests/run_tests.py`. Each was checked for: does retrieval contain the
evidence? is the answer correct, relevant, and faithful to the retrieved context?
is an unsupported question handled gracefully?

## Test questions & results

| # | Type | Question | Outcome |
|---|------|----------|---------|
| 1 | Factual lookup | "Show me black cotton kurtas for a festive occasion" | ⚠️ Over-conservative refusal (see Failure 1) |
| 2 | Factual (attribute) | "What fabric is the InWeave orange kurta with palazzos made of?" | ✅ Correct: "viscose rayon" |
| 3 | Comparison | "Compare a Biba anarkali kurta with a Libas kurta set" | ✅ Full grounded side-by-side |
| 4 | Multi-part + price | "I need a red festive kurta under 3000 and a matching dupatta" | ⚠️ Refused — price not enforced (see Failure 2) |
| 5 | Out-of-dataset | "What is the capital of France?" | ✅ Clean fallback (no LLM call) |
| 6 | Out-of-scope | "Do you have a men's formal shirt?" | ✅ Correctly says not found (catalogue is women's) |
| 7 | Rating / superlative | "What is the best rated anarkali kurta you have?" | ⚠️ "Best" is only over the 5 retrieved (see Limitation) |

### Evidence checks (did retrieval contain what's needed?)

- **Q2 (attribute):** retrieved the exact product *InWeave Women Orange Solid Kurta
  with Palazzos* at distance 0.209 (closest possible). Answer "viscose rayon" is
  verbatim from its context. **Faithful ✅**
- **Q3 (comparison):** retrieved both a *Biba Red Anarkali* (0.368) and a *Libas
  Navy Cotton kurta set* (0.411), tagged **both** (semantic + keyword agreed). The
  comparison (fabric, price, rating, occasion) is drawn entirely from those two
  contexts. **Faithful ✅**
- **Q5 (out-of-dataset):** no product within the 0.72 distance gate → fixed
  fallback returned **without** an LLM call. **Correct refusal ✅**
- **Q6 (men's):** retrieval returned *women's* formal shirts/blazers (Shaftesbury
  London, distances ~0.58–0.60, near the gate). The model did not pretend these
  were men's — it said it couldn't find a men's shirt. **Faithful ✅**

### Qualitative scoring

| Dimension | Assessment |
|-----------|------------|
| **Correctness** | Strong on direct lookups and comparison (Q2, Q3). Weak on constrained/superlative queries (Q1, Q4, Q7). |
| **Relevance** | Retrieved products are on-topic in every case; hybrid adds exact-term precision (brand/fabric). |
| **Faithfulness** | High — no hallucinated products or prices observed; the bot declines rather than inventing. |
| **Unsupported handling** | Excellent — both the hard off-topic case (Q5) and the subtle scope case (Q6) are handled without fabrication. |

---

## Failure case 1 — Over-conservative grounding (false negative)

**Query:** "Show me black cotton kurtas for a festive occasion"
**What happened:** retrieval surfaced relevant black kurtis (*Lakshita Black
Printed Kurti* ₹2695, *Fashion FRICKS Black Solid Kurti* ₹2499, *FASHION DEPTH
Black Embroidered Kurta* ₹1999), but the LLM **refused**, saying it couldn't find
black *cotton festive* kurtas — because none of the retrieved texts stated all
three of "black" + "cotton" + "festive" explicitly.

**Why:** the strict grounding prompt treats a missing attribute (fabric not
stated as cotton) as "insufficient context," so a reasonable partial match is
rejected instead of offered with a caveat.

**How to improve:**
- Soften the prompt to allow *partial* matches: "if some products match most
  criteria, present them and note which attribute is unconfirmed."
- Improve the data: many descriptions do contain fabric, but not always in a
  normalised field — add a dedicated `fabric` metadata column parsed from
  `p_attributes` so fabric can be filtered/confirmed reliably.
- Increase top-k so more candidates are available to satisfy all constraints.

## Failure case 2 — Natural-language constraints not enforced

**Query:** "I need a red festive kurta under 3000 and a matching dupatta"
**What happened:** retrieval returned red festive kurtas, but mostly **over
₹3000** (*Inddus* ₹5899, *Baisacrafts* ₹4450, *Ishin* ₹6199; the only sub-3000
hits were a ₹3198 kurta and a *girls'* lehenga). The "under 3000" and "matching
dupatta" parts were **not applied as hard filters**, so the model couldn't satisfy
the request and refused.

**Why:** the price/colour metadata filters exist in `retrieve()` but are only
wired to the sidebar controls — they are **not parsed from free text**. The
retriever searched on meaning alone and ignored the numeric budget.

**How to improve:**
- Add a lightweight **query-understanding step**: extract price ceilings
  (`under 3000` → `price_max=3000`), colours, and occasion from the text and pass
  them into the existing metadata filters.
- Handle multi-part requests by **decomposing** them (query-decomposition RAG):
  retrieve "red festive kurta under 3000" and "matching red dupatta" separately,
  then combine — rather than retrieving once for the whole sentence.

## Additional limitation — Superlatives need aggregation, not top-k

**Query:** "What is the best rated anarkali kurta you have?"
The bot returned *KALINI Pink Bandhani Anarkali, 4.4/5* — but that is only the
best-rated among the **5 retrieved** products, not the whole catalogue. RAG
retrieves semantically similar items; it does not compute global max/min. A true
"best rated overall" answer needs a metadata query (sort by `avg_rating` with a
minimum `rating_count`), which is a different operation from semantic retrieval.
Worth noting in the demo as a known boundary of the approach.
