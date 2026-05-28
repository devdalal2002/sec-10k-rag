"""
src/chunk.py — Two chunking strategies for 10-K filings.

Strategy A (recursive): RecursiveCharacterTextSplitter across full filing text.
  Section attribution determined post-hoc by character offset.
Strategy B (section_aware): Splits within sections only. Chunks never cross boundaries.

Output:
  data/chunks/recursive.jsonl
  data/chunks/section_aware.jsonl
"""

import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

PROCESSED_DIR = Path("data/processed")
CHUNKS_DIR = Path("data/chunks")

COMPANIES = ["AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "JPM", "GS", "WMT", "TSLA"]
YEARS = [2022, 2023, 2024]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_SECTION_CHARS = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)


def _section_for_pos(pos: int, offsets: list) -> tuple[str, str]:
    for start, end, sid, stitle in offsets:
        if start <= pos < end:
            return sid, stitle
    return offsets[-1][2], offsets[-1][3]


def chunk_recursive(filing: dict) -> list[dict]:
    sections = [s for s in filing["sections"] if s["char_count"] >= MIN_SECTION_CHARS]
    if not sections:
        return []

    # Concatenate all sections, track byte offsets per section
    SEP = "\n\n"
    parts, offsets, pos = [], [], 0
    for i, s in enumerate(sections):
        offsets.append((pos, pos + len(s["text"]), s["section_id"], s["section_title"]))
        parts.append(s["text"])
        pos += len(s["text"])
        if i < len(sections) - 1:
            parts.append(SEP)
            pos += len(SEP)

    full_text = "".join(parts)
    raw_chunks = _splitter.split_text(full_text)

    chunks = []
    search_from = 0
    for idx, text in enumerate(raw_chunks):
        # Forward scan — amortized O(n) across all chunks
        chunk_pos = full_text.find(text, search_from)
        if chunk_pos == -1:
            chunk_pos = full_text.find(text)

        sid, stitle = _section_for_pos(max(chunk_pos, 0), offsets)

        chunks.append({
            "chunk_id": f"{filing['ticker']}_{filing['fiscal_year']}_recursive_{idx:04d}",
            "ticker": filing["ticker"],
            "fiscal_year": filing["fiscal_year"],
            "filing_date": filing["filing_date"],
            "strategy": "recursive",
            "section_id": sid,
            "section_title": stitle,
            "text": text,
            "char_count": len(text),
            "chunk_index_in_filing": idx,
        })

        if chunk_pos >= 0:
            search_from = chunk_pos + max(1, len(text) - CHUNK_OVERLAP)

    return chunks


def chunk_section_aware(filing: dict) -> list[dict]:
    chunks = []
    global_idx = 0

    for section in filing["sections"]:
        if section["char_count"] < MIN_SECTION_CHARS:
            continue

        text = section["text"]
        sid = section["section_id"]
        stitle = section["section_title"]

        raw = [text] if section["char_count"] <= CHUNK_SIZE else _splitter.split_text(text)

        for raw_chunk in raw:
            # Non-negotiable: every chunk must be a substring of this section's text
            assert text.find(raw_chunk[:100]) != -1, (
                f"Cross-section chunk detected in {sid} "
                f"({filing['ticker']} {filing['fiscal_year']})"
            )
            chunks.append({
                "chunk_id": f"{filing['ticker']}_{filing['fiscal_year']}_section_aware_{global_idx:04d}",
                "ticker": filing["ticker"],
                "fiscal_year": filing["fiscal_year"],
                "filing_date": filing["filing_date"],
                "strategy": "section_aware",
                "section_id": sid,
                "section_title": stitle,
                "text": raw_chunk,
                "char_count": len(raw_chunk),
                "chunk_index_in_filing": global_idx,
            })
            global_idx += 1

    return chunks


def main():
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    rec_path = CHUNKS_DIR / "recursive.jsonl"
    sec_path = CHUNKS_DIR / "section_aware.jsonl"

    rec_total = sec_total = 0

    with open(rec_path, "w", encoding="utf-8") as rf, \
         open(sec_path, "w", encoding="utf-8") as sf:

        for ticker in COMPANIES:
            for year in YEARS:
                path = PROCESSED_DIR / ticker / f"{year}.json"
                if not path.exists():
                    print(f"  [skip] {ticker} {year}")
                    continue

                with open(path, encoding="utf-8") as f:
                    filing = json.load(f)

                rec = chunk_recursive(filing)
                sec = chunk_section_aware(filing)

                for c in rec:
                    rf.write(json.dumps(c, ensure_ascii=False) + "\n")
                for c in sec:
                    sf.write(json.dumps(c, ensure_ascii=False) + "\n")

                rec_total += len(rec)
                sec_total += len(sec)
                print(f"  {ticker} {year}: {len(rec)} recursive, {len(sec)} section_aware")

    print(f"\nrecursive.jsonl:     {rec_total:,} chunks")
    print(f"section_aware.jsonl: {sec_total:,} chunks")


if __name__ == "__main__":
    main()
