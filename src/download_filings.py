"""
Download 5 SEC 10-K filings (Apple, Microsoft, Nvidia, Meta, Google).
Saves HTML files to data/raw/.

Usage:
    python src/download_filings.py

SEC EDGAR requires a real User-Agent string with your contact info.
Edit the USER_AGENT constant below before running.
"""

import os
import time
import requests

USER_AGENT = "RAG Project student durva.aageyseright@gmail.com"

FILINGS = [
    {
        "company": "Apple",
        "ticker": "AAPL",
        "cik": "0000320193",
        "accession": "0000320193-24-000123",
        "filename": "aapl-20240928.htm",
    },
    {
        "company": "Microsoft",
        "ticker": "MSFT",
        "cik": "0000789019",
        "accession": "0000789019-24-000044",
        "filename": "msft-20240630.htm",
    },
    {
        "company": "Nvidia",
        "ticker": "NVDA",
        "cik": "0001045810",
        "accession": "0001045810-24-000139",
        "filename": "nvda-20240128.htm",
    },
    {
        "company": "Meta",
        "ticker": "META",
        "cik": "0001326801",
        "accession": "0001326801-24-000012",
        "filename": "meta-20231231.htm",
    },
    {
        "company": "Google (Alphabet)",
        "ticker": "GOOGL",
        "cik": "0001652044",
        "accession": "0001652044-24-000022",
        "filename": "goog-20231231.htm",
    },
]

BASE_URL = "https://www.sec.gov/Archives/edgar/data"


def accession_path(accession: str) -> str:
    return accession.replace("-", "")


def download_filing(filing: dict, out_dir: str) -> bool:
    cik_clean = filing["cik"].lstrip("0")
    acc_path = accession_path(filing["accession"])
    url = f"{BASE_URL}/{cik_clean}/{acc_path}/{filing['filename']}"
    out_path = os.path.join(out_dir, f"{filing['ticker'].lower()}_10k.htm")

    if os.path.exists(out_path):
        print(f"  [skip] {filing['company']} already downloaded")
        return True

    headers = {"User-Agent": USER_AGENT}
    print(f"  Downloading {filing['company']} from {url}")
    try:
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
        size_kb = len(resp.content) // 1024
        print(f"  [ok] {filing['company']} -> {out_path} ({size_kb} KB)")
        return True
    except requests.HTTPError as e:
        print(f"  [error] {filing['company']}: HTTP {e.response.status_code}")
        print(f"          Try navigating to https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={filing['cik']}&type=10-K")
        return False
    except Exception as e:
        print(f"  [error] {filing['company']}: {e}")
        return False


def main():
    out_dir = os.path.join("data", "raw")
    os.makedirs(out_dir, exist_ok=True)

    print("SEC 10-K Downloader")
    print("=" * 50)
    print(f"Output directory: {out_dir}\n")

    successes = 0
    for filing in FILINGS:
        ok = download_filing(filing, out_dir)
        if ok:
            successes += 1
        time.sleep(0.5)  # SEC rate limit: be polite

    print(f"\n{successes}/{len(FILINGS)} filings downloaded to {out_dir}/")
    if successes < len(FILINGS):
        print("\nFor failed filings, visit SEC EDGAR manually:")
        print("  https://efts.sec.gov/LATEST/search-index?q=%2210-K%22&dateRange=custom&startdt=2023-01-01&enddt=2024-12-31&forms=10-K")
        print("Download the .htm file and save it to data/raw/<ticker>_10k.htm")


if __name__ == "__main__":
    main()
