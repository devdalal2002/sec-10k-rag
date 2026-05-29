"""
src/extract.py - Convert 10-K HTML filings to structured JSON.

Reads:  data/raw/{ticker}/{year}.html
Writes: data/processed/{ticker}/{year}.json
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup, Comment

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MANIFEST_PATH = RAW_DIR / "manifest.csv"

REQUIRED = {"item_1", "item_1a", "item_7", "item_7a", "item_8", "item_9a"}

KNOWN_SECTIONS = {
    "1":  ("item_1",  "Business"),
    "1a": ("item_1a", "Risk Factors"),
    "1b": ("item_1b", "Cybersecurity"),
    "2":  ("item_2",  "Properties"),
    "3":  ("item_3",  "Legal Proceedings"),
    "4":  ("item_4",  "Mine Safety Disclosures"),
    "5":  ("item_5",  "Market Information"),
    "6":  ("item_6",  "Reserved"),
    "7":  ("item_7",  "MD&A"),
    "7a": ("item_7a", "Quantitative Risk"),
    "8":  ("item_8",  "Financial Statements"),
    "9":  ("item_9",  "Changes in Disagreements"),
    "9a": ("item_9a", "Controls and Procedures"),
    "9b": ("item_9b", "Other Information"),
    "9c": ("item_9c", "Insider Trading"),
    "10": ("item_10", "Directors and Officers"),
    "11": ("item_11", "Executive Compensation"),
    "12": ("item_12", "Security Ownership"),
    "13": ("item_13", "Certain Relationships"),
    "14": ("item_14", "Principal Accountant Fees"),
    "15": ("item_15", "Exhibits"),
    "16": ("item_16", "Form 10-K Summary"),
}

# Matches "Item 1A.", "ITEM 7A—", "item 9a:", etc.
ITEM_RE = re.compile(
    r'(?:^|\n)[ \t]*item\s+(\d+\s*[a-z]?)[ \t]*[.\-—–:]?[ \t]*([^\n]{0,120})',
    re.IGNORECASE | re.MULTILINE,
)


def clean_html(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    # Strip XBRL/namespace tags, keep their text content
    raw = re.sub(r'<ix:header[^>]*>.*?</ix:header>', '', raw,
                 flags=re.IGNORECASE | re.DOTALL)
    raw = re.sub(r'</?[a-zA-Z][a-zA-Z0-9]*:[a-zA-Z][^>]*>', '', raw)

    soup = BeautifulSoup(raw, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    for node in soup.find_all(string=lambda s: isinstance(s, Comment)):
        node.extract()

    # Convert tables innermost-first to avoid nesting issues
    for table in reversed(soup.find_all("table")):
        lines = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            row = " | ".join(c for c in cells if c.strip())
            if row:
                lines.append(row)
        if lines:
            table.replace_with("\n" + "\n".join(lines) + "\n")
        else:
            table.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_sections(text: str) -> tuple[list[dict], list[str]]:
    warnings: list[str] = []

    candidates = []
    for m in ITEM_RE.finditer(text):
        item_key = re.sub(r'\s+', '', m.group(1)).lower()  # "1a", "7", "9a" …
        title_raw = m.group(2).strip().rstrip('.')
        if item_key not in KNOWN_SECTIONS:
            continue
        candidates.append({
            "item": item_key,
            "title": title_raw,
            "pos": m.start(),
        })

    if not candidates:
        warnings.append("no item headers found")
        return [], warnings

    # Content between this candidate and the next
    for i, c in enumerate(candidates):
        end = candidates[i + 1]["pos"] if i + 1 < len(candidates) else len(text)
        c["body"] = text[c["pos"]:end]
        c["body_len"] = len(c["body"])

    # Per item: keep only the candidate with the most content (skips TOC entries)
    best: dict[str, dict] = defaultdict(lambda: {"body_len": -1})
    for c in candidates:
        if c["body_len"] > best[c["item"]]["body_len"]:
            best[c["item"]] = c

    # Sort by document position
    ordered = sorted(best.values(), key=lambda c: c["pos"])

    sections = []
    for order, c in enumerate(ordered):
        section_id, default_title = KNOWN_SECTIONS[c["item"]]

        # Strip the header line itself from the body
        body = c["body"]
        nl = body.find('\n')
        body = body[nl:].strip() if nl != -1 else body

        title = c["title"] if len(c["title"]) > 4 else default_title

        sections.append({
            "section_id": section_id,
            "section_title": title,
            "text": body,
            "char_count": len(body),
            "order": order,
        })

    found_ids = {s["section_id"] for s in sections}
    for req in sorted(REQUIRED):
        if req not in found_ids:
            warnings.append(f"{req} not detected")

    return sections, warnings


def load_manifest() -> list[dict]:
    rows = []
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["file_path"]:
                rows.append(row)
    return rows


def main():
    manifest = load_manifest()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ok = 0
    for row in manifest:
        ticker, year = row["ticker"], row["fiscal_year"]
        out_path = PROCESSED_DIR / ticker / f"{year}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"  {ticker} {year} ...", end=" ", flush=True)
        try:
            text = clean_html(Path(row["file_path"]))
            sections, warnings = extract_sections(text)
            result = {
                "ticker": ticker,
                "fiscal_year": int(year),
                "filing_date": row["filing_date"],
                "sections": sections,
                "total_chars": sum(s["char_count"] for s in sections),
                "extraction_warnings": warnings,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"{len(sections)} sections, {result['total_chars']:,} chars, "
                  f"{len(warnings)} warnings")
            ok += 1
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nDone: {ok}/{len(manifest)} -> {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
