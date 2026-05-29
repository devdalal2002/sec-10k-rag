"""
src/chunk_report.py - Comparison table for both chunking strategies.
"""

import json
import statistics
from collections import Counter
from pathlib import Path

CHUNKS_DIR = Path("data/chunks")


def load_jsonl(path: Path) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def print_stats(label: str, chunks: list[dict]) -> None:
    sizes = [c["char_count"] for c in chunks]
    avg = statistics.mean(sizes)
    std = statistics.stdev(sizes) if len(sizes) > 1 else 0
    print(f"{label:<18} {len(chunks):>12,}   {avg:>9,.0f}   {min(sizes):>5}   {max(sizes):>5}   {std:>6,.0f}")


def main():
    rec = load_jsonl(CHUNKS_DIR / "recursive.jsonl")
    sec = load_jsonl(CHUNKS_DIR / "section_aware.jsonl")

    print(f"\n{'Strategy':<18} {'Total Chunks':>12}   {'Avg Chars':>9}   {'Min':>5}   {'Max':>5}   {'Std':>6}")
    print("-" * 72)
    print_stats("recursive", rec)
    print_stats("section_aware", sec)

    aapl = [c for c in sec if c["ticker"] == "AAPL" and c["fiscal_year"] == 2023]
    counts = Counter(c["section_id"] for c in aapl)
    print(f"\nAAPL 2023 section_aware per-section chunk counts:")
    for sid in sorted(counts):
        print(f"  {sid}: {counts[sid]}")


if __name__ == "__main__":
    main()
