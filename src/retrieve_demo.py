"""
src/retrieve_demo.py - Runs 3 test queries x 4 configs x 2 collections.

Prints top-3 per block and reports:
  - Rerank latency across all reranked runs
  - Q1 honest comparison: dense vs hybrid_rerank_filter
  - Cross-collection overlap for Q1 (recursive vs section_aware)
"""

import time
from retrieve import retrieve, _parse_entities, _build_where, _get_index

COLLECTIONS = ["sec_recursive", "sec_section_aware"]
CONFIGS = ["dense", "hybrid", "hybrid_rerank", "hybrid_rerank_filter"]
TOP_K = 5
SHOW_N = 3

QUERIES = [
    {
        "label": "Q1 - Single entity + year (filter should help)",
        "text": "What was Nvidia's data center revenue in fiscal 2024?",
    },
    {
        "label": "Q2 - Multi-entity (filter must not over-restrict)",
        "text": "Compare R&D spending across Apple, Microsoft, and Google",
    },
    {
        "label": "Q3 - No entity (should search broadly, hit item_1c/item_1a)",
        "text": "What cybersecurity risks are disclosed?",
    },
]


def fmt_score(config: str, score: float) -> str:
    if config in ("hybrid_rerank", "hybrid_rerank_filter"):
        return f"{score:.4f} (logit)"
    return f"{score:.4f}"


def print_block(config: str, results: list[dict], lat_note: str) -> None:
    print(f"\n  [{config}]  {lat_note}")
    for i, r in enumerate(results[:SHOW_N], 1):
        score_str = fmt_score(config, r["score"])
        print(f"    [{i}] {r['chunk_id']}")
        print(f"         section={r['section_id']}  ticker={r['ticker']}"
              f"  year={r['fiscal_year']}  score={score_str}")
        print(f"         {r['text'][:180].replace(chr(10), ' ')}")


def main() -> None:
    rerank_latencies: list[float] = []
    # q1_results[collection][config] -> list[dict]
    q1_results: dict[str, dict[str, list[dict]]] = {}

    for q in QUERIES:
        tickers, years = _parse_entities(q["text"])
        where = _build_where(tickers, years)
        print(f"\n{'='*70}")
        print(f"{q['label']}")
        print(f"Query: \"{q['text']}\"")
        print(f"  Detected: tickers={tickers}  years={years}  filter={where}")

        for collection in COLLECTIONS:
            print(f"\n  -- Collection: {collection} --")
            coll_results: dict[str, list[dict]] = {}

            for config in CONFIGS:
                t0 = time.perf_counter()
                results = retrieve(q["text"], collection, config, top_k=TOP_K)
                wall_ms = (time.perf_counter() - t0) * 1000

                lat_note = ""
                if results and "rerank_latency_ms" in results[0]:
                    lat = results[0]["rerank_latency_ms"]
                    rerank_latencies.append(lat)
                    lat_note = f"rerank={lat:.0f}ms  total={wall_ms:.0f}ms"
                else:
                    lat_note = f"total={wall_ms:.0f}ms"

                coll_results[config] = results
                print_block(config, results, lat_note)

            if q["label"].startswith("Q1"):
                q1_results[collection] = coll_results

        # Q1-specific summaries after both collections
        if q["label"].startswith("Q1") and len(q1_results) == 2:
            print(f"\n  --- Q1 within-collection: dense vs hybrid_rerank_filter ---")
            for coll in COLLECTIONS:
                dense_ids = {r["chunk_id"] for r in q1_results[coll]["dense"]}
                filt_ids  = {r["chunk_id"] for r in q1_results[coll]["hybrid_rerank_filter"]}
                shared = dense_ids & filt_ids
                d_secs = [r["section_id"] for r in q1_results[coll]["dense"][:SHOW_N]]
                f_secs = [r["section_id"] for r in q1_results[coll]["hybrid_rerank_filter"][:SHOW_N]]
                print(f"  {coll}:")
                print(f"    dense top-{SHOW_N} sections:               {d_secs}")
                print(f"    hybrid_rerank_filter top-{SHOW_N} sections: {f_secs}")
                print(f"    chunks in common (top-{TOP_K}): {len(shared)}/{TOP_K}")

            print(f"\n  --- Q1 cross-collection chunk overlap (hybrid_rerank_filter) ---")
            rec_ids  = {r["chunk_id"] for r in q1_results["sec_recursive"]["hybrid_rerank_filter"]}
            sec_ids  = {r["chunk_id"] for r in q1_results["sec_section_aware"]["hybrid_rerank_filter"]}
            # chunk_ids differ by strategy label, compare by position-independent content
            rec_texts = {r["text"][:80] for r in q1_results["sec_recursive"]["hybrid_rerank_filter"]}
            sec_texts = {r["text"][:80] for r in q1_results["sec_section_aware"]["hybrid_rerank_filter"]}
            shared_texts = rec_texts & sec_texts
            print(f"  recursive chunk_ids:    {sorted(rec_ids)}")
            print(f"  section_aware chunk_ids: {sorted(sec_ids)}")
            print(f"  Texts in common (top-{TOP_K} of each): {len(shared_texts)}/{TOP_K}")

    # BM25 alignment proof
    print(f"\n{'='*70}")
    print("BM25 alignment verification:")
    for coll in COLLECTIONS:
        idx = _get_index(coll)
        sample = list(idx._chunks.keys())[:2]
        all_aligned = all(
            ("section_aware" in cid) == ("section_aware" in coll)
            for cid in list(idx._chunks.keys())
        )
        print(f"  {coll}: {len(idx._chunks):,} chunks, sample={sample}, aligned={all_aligned}")

    if rerank_latencies:
        avg = sum(rerank_latencies) / len(rerank_latencies)
        print(f"\nRerank latency ({len(rerank_latencies)} runs): "
              f"avg={avg:.0f}ms  min={min(rerank_latencies):.0f}ms  "
              f"max={max(rerank_latencies):.0f}ms")


if __name__ == "__main__":
    main()
