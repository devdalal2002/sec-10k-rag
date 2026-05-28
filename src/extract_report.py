"""
src/extract_report.py — Print extraction summary table for all 30 filings.
"""

import json
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
COMPANIES = ["AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "JPM", "GS", "WMT", "TSLA"]
YEARS = [2022, 2023, 2024]


def main():
    print(f"{'Ticker':<8}{'Year':<6}{'Sections':<10}{'Total Chars':<16}Warnings")
    print("-" * 70)

    for ticker in COMPANIES:
        for year in YEARS:
            path = PROCESSED_DIR / ticker / f"{year}.json"
            if not path.exists():
                print(f"{ticker:<8}{year:<6}{'MISSING'}")
                continue

            with open(path, encoding="utf-8") as f:
                d = json.load(f)

            w = d["extraction_warnings"]
            warn_str = "0" if not w else f"{len(w)} ({'; '.join(w)})"
            print(f"{ticker:<8}{str(year):<6}{len(d['sections']):<10}"
                  f"{d['total_chars']:>12,}    {warn_str}")


if __name__ == "__main__":
    main()
