"""
src/generate_demo.py — End-to-end: retrieve -> generate -> cited answer.

Three queries:
  Q1: Answerable, numerical  — Nvidia data center revenue FY2024
  Q2: Answerable, narrative  — Microsoft cybersecurity disclosures
  Q3: Refusal test           — Nvidia cryptocurrency mining revenue FY2024
                               (should not be in filings; test for hallucination vs refusal)
"""

import textwrap
import time
from retrieve import retrieve
from generate import generate_answer, SYSTEM_PROMPT, DEFAULT_MODEL

COLLECTION  = "sec_recursive"
CONFIG      = "hybrid_rerank_filter"
TOP_K       = 5
MODEL       = DEFAULT_MODEL

QUERIES = [
    {
        "label": "Q1 — Answerable, numerical",
        "query": "What was Nvidia's data center revenue in fiscal 2024?",
    },
    {
        "label": "Q2 — Answerable, narrative",
        "query": "What cybersecurity risks does Microsoft disclose?",
    },
    {
        "label": "Q3 — Refusal test (likely not in filings)",
        "query": "What was Nvidia's cryptocurrency mining revenue in 2024?",
    },
]

WRAP = 90


def hr(char: str = "=", n: int = 70) -> str:
    return char * n


def run() -> None:
    print(f"Model: {MODEL}  |  Collection: {COLLECTION}  |  Config: {CONFIG}")
    print(f"\nSystem prompt:\n{hr('-')}")
    for line in SYSTEM_PROMPT.splitlines():
        print(f"  {line}")
    print(hr("-"))

    for q in QUERIES:
        print(f"\n{hr()}")
        print(f"{q['label']}")
        print(f"Query: \"{q['query']}\"")

        # --- Retrieve ---
        t_ret = time.perf_counter()
        chunks = retrieve(q["query"], COLLECTION, CONFIG, top_k=TOP_K)
        ret_ms = (time.perf_counter() - t_ret) * 1000

        print(f"\nChunks fed in (top-{len(chunks)}, retrieval={ret_ms:.0f}ms):")
        for i, c in enumerate(chunks, 1):
            print(f"  [{i}] {c['chunk_id']}  section={c['section_id']}"
                  f"  score={c['score']:.4f}")

        # --- Generate ---
        result = generate_answer(q["query"], chunks, model=MODEL)

        print(f"\nAnswer (generation={result['generation_ms']:.0f}ms):")
        wrapped = textwrap.fill(result["answer"], width=WRAP,
                                initial_indent="  ", subsequent_indent="  ")
        print(wrapped)

        print(f"\nCitations ({len(result['citations'])} chunk(s)):")
        if result["citations"]:
            for cit in result["citations"]:
                print(f"  {cit['chunk_id']}  "
                      f"ticker={cit['ticker']}  year={cit['fiscal_year']}  "
                      f"section={cit['section_id']}")
        else:
            print("  (none cited)")

        if result["hallucinated_citations"]:
            print(f"\n  WARNING: model cited out-of-range indices "
                  f"{result['hallucinated_citations']} — not in the {TOP_K} chunks passed")

        # Q3 explicit refusal check
        if q["label"].startswith("Q3"):
            refusal_phrase = "The provided filings do not contain this information"
            refused = refusal_phrase.lower() in result["answer"].lower()
            print(f"\n  Refusal check: {'PASSED — model refused to answer' if refused else 'FAILED — model did not use the refusal phrase'}")
            print(f"  Raw answer quoted below for scrutiny:")
            print(f"  ---")
            for line in result["answer"].splitlines():
                print(f"  {line}")
            print(f"  ---")


if __name__ == "__main__":
    run()
