"""
Download 10-K filings for 10 companies x 3 fiscal years (2022, 2023, 2024).
Saves to data/raw/{ticker}/{fiscal_year}.html and writes data/raw/manifest.csv.

Usage:
    python src/download_filings.py
"""

import csv
import time
from pathlib import Path

import requests

USER_AGENT = "Dev Dalal dalaldh2002@gmail.com"
HEADERS = {"User-Agent": USER_AGENT}

COMPANIES = [
    {"ticker": "AAPL",  "company": "Apple",          "cik": "0000320193"},
    {"ticker": "MSFT",  "company": "Microsoft",       "cik": "0000789019"},
    {"ticker": "NVDA",  "company": "Nvidia",          "cik": "0001045810"},
    {"ticker": "META",  "company": "Meta",            "cik": "0001326801"},
    {"ticker": "GOOGL", "company": "Alphabet",        "cik": "0001652044"},
    {"ticker": "AMZN",  "company": "Amazon",          "cik": "0001018724"},
    {"ticker": "JPM",   "company": "JPMorgan Chase",  "cik": "0000019617"},
    {"ticker": "GS",    "company": "Goldman Sachs",   "cik": "0000886982"},
    {"ticker": "WMT",   "company": "Walmart",         "cik": "0000104169"},
    {"ticker": "TSLA",  "company": "Tesla",           "cik": "0001318605"},
]

TARGET_YEARS = [2022, 2023, 2024]

RAW_DIR = Path("data/raw")
MANIFEST_PATH = RAW_DIR / "manifest.csv"
MANIFEST_COLS = [
    "ticker", "company_name", "fiscal_year",
    "filing_date", "url", "file_path", "file_size_bytes",
]


def _extract_10ks(filing_block: dict) -> list[dict]:
    out = []
    for i, form in enumerate(filing_block["form"]):
        if form == "10-K":
            out.append({
                "accession":   filing_block["accessionNumber"][i],
                "filing_date": filing_block["filingDate"][i],
                "report_date": filing_block["reportDate"][i],
                "primary_doc": filing_block["primaryDocument"][i],
            })
    return out


def fetch_10k_filings(cik: str) -> list[dict]:
    r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    time.sleep(0.15)

    data = r.json()
    filings = _extract_10ks(data["filings"]["recent"])

    found_years = {f["report_date"][:4] for f in filings}
    missing_years = [y for y in TARGET_YEARS if str(y) not in found_years]
    if not missing_years:
        return filings

    # Paginate - filing pages are ordered newest-first, each covering a date range
    for page in data["filings"].get("files", []):
        if not missing_years:
            break
        page_from = page["filingFrom"]
        page_to   = page["filingTo"]
        # A 10-K with reportDate in year Y is typically filed between Y-01-01 and Y+1-06-30
        relevant = any(
            page_to >= f"{year}-01-01" and page_from <= f"{year + 1}-06-30"
            for year in missing_years
        )
        if not relevant:
            if page_to < f"{min(missing_years)}-01-01":
                break  # pages are newest-first; nothing older will help
            continue

        pr = requests.get(f"https://data.sec.gov/submissions/{page['name']}",
                          headers=HEADERS, timeout=30)
        pr.raise_for_status()
        time.sleep(0.15)

        page_filings = _extract_10ks(pr.json())
        filings.extend(page_filings)
        found_years |= {f["report_date"][:4] for f in page_filings}
        missing_years = [y for y in missing_years if str(y) not in found_years]

    return filings


def find_for_year(filings: list[dict], year: int) -> dict | None:
    matches = [f for f in filings if f["report_date"].startswith(str(year))]
    if not matches:
        return None
    return sorted(matches, key=lambda f: f["report_date"])[-1]


def build_url(cik: str, filing: dict) -> str:
    cik_int = str(int(cik))
    acc_clean = filing["accession"].replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{filing['primary_doc']}"


def download(url: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, headers=HEADERS, timeout=120, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
    time.sleep(0.15)
    return dest.stat().st_size


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    total_have = 0
    missing = []

    for co in COMPANIES:
        ticker, company, cik = co["ticker"], co["company"], co["cik"]
        print(f"\n{ticker} ({company})")

        try:
            filings = fetch_10k_filings(cik)
        except Exception as e:
            print(f"  [error] EDGAR lookup failed: {e}")
            for year in TARGET_YEARS:
                missing.append(f"{ticker} FY{year}")
                rows.append({"ticker": ticker, "company_name": company,
                              "fiscal_year": year, "filing_date": "",
                              "url": "", "file_path": "", "file_size_bytes": ""})
            continue

        for year in TARGET_YEARS:
            dest = RAW_DIR / ticker / f"{year}.html"

            filing = find_for_year(filings, year)
            if not filing:
                print(f"  [missing] FY{year} - no 10-K with reportDate in {year}")
                missing.append(f"{ticker} FY{year}")
                rows.append({"ticker": ticker, "company_name": company,
                              "fiscal_year": year, "filing_date": "",
                              "url": "", "file_path": "", "file_size_bytes": ""})
                continue

            url = build_url(cik, filing)

            if dest.exists():
                size = dest.stat().st_size
                print(f"  [skip]  FY{year} already on disk ({size // 1024} KB)")
                rows.append({"ticker": ticker, "company_name": company,
                              "fiscal_year": year, "filing_date": filing["filing_date"],
                              "url": url, "file_path": str(dest),
                              "file_size_bytes": size})
                total_have += 1
                continue

            try:
                size = download(url, dest)
                print(f"  [ok]    FY{year} -> {dest} ({size // 1024} KB)")
                rows.append({"ticker": ticker, "company_name": company,
                              "fiscal_year": year, "filing_date": filing["filing_date"],
                              "url": url, "file_path": str(dest),
                              "file_size_bytes": size})
                total_have += 1
            except Exception as e:
                print(f"  [error] FY{year}: {e}")
                missing.append(f"{ticker} FY{year}")
                rows.append({"ticker": ticker, "company_name": company,
                              "fiscal_year": year, "filing_date": filing["filing_date"],
                              "url": url, "file_path": "", "file_size_bytes": ""})

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDownloaded {total_have} of 30 filings.")
    if missing:
        print(f"Missing: {missing}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
