"""
src/retrieve_demo.py — Runs 3 test queries x 4 configs and prints top-3 results.

Also reports:
  - Average rerank latency across hybrid_rerank / hybrid_rerank_filter runs
  - Honest comparison of dense vs hybrid_rerank_filter for query 1
"""

import time
from retrieve import retrieve, _parse_entities, _build_where

COLLECTION = "sec_recursive"
CONFIGS = ["dense", "hybrid", "hybrid_rerank", "hybrid_rerank_filter"]
TOP_K = 5
SHOW_N = 3

QUERIES = [
    {
        "label": "Q1 — Single entity + year (filter should help)",
        "text": "What was Nvidia's data center revenue in fiscal 2024?",
    },
    {
        "label": "Q2 — Multi-entity (filter must not over-restrict)",
        "text": "Compare R&D spending across Apple, Microsoft, and Google",
    },
    {
        "label": "Q3 — No entity (should search broadly, hit item_1c/item_1a)",
        "text": "What cybersecurity risks are disclosed?",
    },
]


def fmt_score(config: str, score: float) -> str:
    """Dense/hybrid scores are cosine/RRF; rerank scores are logits — label them."""
    if config in ("hybrid_rerank", "hybrid_rerank_filter"):
        return f"{score:.4f} (logit)"
    return f"{score:.4f}"


def print_results(label: str, config: str, results: list[dict]) -> None:
    print(f"\n  [{config}]")
    for i, r in enumerate(results[:SHOW_N], 1):
        score_str = fmt_score(config, r["score"])
        print(f"    [{i}] {r['chunk_id']}")
        print(f"         section={r['section_id']}  ticker={r['ticker']}"
              f"  year={r['fiscal_year']}  score={score_str}")
        print(f"         {r['text'][:180].replace(chr(10), ' ')}")


def main() -> None:
    rerank_latencies: list[float] = []

    for q in QUERIES:
        print(f"\n{'='*70}")
        print(f"{q['label']}")
        print(f"Query: \"{q['text']}\"")

        tickers, years = _parse_entities(q["text"])
        where = _build_where(tickers, years)
        print(f"  Detected: tickers={tickers}  years={years}  filter={where}")

        query_results: dict[str, list[dict]] = {}

        for config in CONFIGS:
            t0 = time.perf_counter()
            results = retrieve(q["text"], COLLECTION, config, top_k=TOP_K)
            wall_ms = (time.perf_counter() - t0) * 1000

            query_results[config] = results
            latency_note = ""
            if results and "rerank_latency_ms" in results[0]:
                lat = results[0]["rerank_latency_ms"]
                rerank_latencies.append(lat)
                latency_note = f"  (rerank={lat:.0f}ms, total={wall_ms:.0f}ms)"
            else:
                latency_note = f"  (total={wall_ms:.0f}ms)"

            print_results(q["label"], config, results)
            print(f"    {latency_note.strip()}")

        # Q1-specific honest comparison
        if q["label"].startswith("Q1"):
            print(f"\n  --- Q1 dense vs hybrid_rerank_filter comparison ---")
            dense_ids = {r["chunk_id"] for r in query_results["dense"]}
            filt_ids  = {r["chunk_id"] for r in query_results["hybrid_rerank_filter"]}
            shared     = dense_ids & filt_ids
            dense_secs = [r["section_id"] for r in query_results["dense"][:SHOW_N]]
            filt_secs  = [r["section_id"] for r in query_results["hybrid_rerank_filter"][:SHOW_N]]
            print(f"  dense top-{SHOW_N} sections:               {dense_secs}")
            print(f"  hybrid_rerank_filter top-{SHOW_N} sections: {filt_secs}")
            print(f"  chunks in common (top-{TOP_K}): {len(shared)}/{TOP_K}")

    if rerank_latencies:
        avg = sum(rerank_latencies) / len(rerank_latencies)
        print(f"\n{'='*70}")
        print(f"Rerank latency across {len(rerank_latencies)} runs: "
              f"avg={avg:.0f}ms  min={min(rerank_latencies):.0f}ms  "
              f"max={max(rerank_latencies):.0f}ms")


if __name__ == "__main__":
    main()
