"""
Download 5 SEC 10-K filings for the RAG corpus.

SEC requires a User-Agent header with contact info. Filings are public,
no auth needed. URLs below are stable EDGAR links to recent 10-Ks.
"""
import os
import requests
from pathlib import Path

# 5 recent 10-K filings, mix of sectors for variety
FILINGS = {
    "AAPL_2024": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm",
    "MSFT_2024": "https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm",
    "NVDA_2024": "https://www.sec.gov/Archives/edgar/data/1045810/000104581024000316/nvda-20240128.htm",
    "META_2024": "https://www.sec.gov/Archives/edgar/data/1326801/000132680124000012/meta-20231231.htm",
    "GOOGL_2024": "https://www.sec.gov/Archives/edgar/data/1652044/000165204424000022/goog-20231231.htm",
}

HEADERS = {
    "User-Agent": "Dev Dalal dalaldh2002@gmail.com",
    "Accept": "text/html",
}

def main():
    out_dir = Path(__file__).parent.parent / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ticker, url in FILINGS.items():
        out_path = out_dir / f"{ticker}.html"
        if out_path.exists():
            print(f"[skip] {ticker} already downloaded")
            continue
        print(f"[fetch] {ticker} from {url}")
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        print(f"[saved] {out_path} ({len(r.content):,} bytes)")

if __name__ == "__main__":
    main()
