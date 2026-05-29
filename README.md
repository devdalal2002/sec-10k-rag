# SEC 10-K RAG System

Hybrid retrieval-augmented generation over 30 SEC 10-K annual reports (10 companies x 3 fiscal
years, ~22K chunks). Evaluated against a hand-verified 65-question benchmark across 8 retrieval
configurations.

**100% recall@10 with named-entity metadata filtering.**

---

## Four Findings

### 1. Metadata filtering is the dominant lever (+10-15 pts recall@5)

Named-entity filtering - extracting ticker and fiscal year from the query and narrowing the
candidate pool before reranking - lifted recall@5 from ~84% to 94.9% and recall@10 to 100%.
The mechanism: filtering reduces the retrieval problem from "find the right chunk in 22K" to
"find the right chunk among ~300 same-company/year chunks," eliminating the main source of
false positives before any ranking happens.

### 2. Reranking alone decreases recall@5 (−5 points)

Adding bge-reranker-base without metadata filtering dropped recall@5 from 83.1% to 79.7% while
improving recall@10 from 93.2% to 94.9%. The reranker moved relevant chunks from positions 1-5
into positions 6-10. Most likely cause: the cross-encoder is trained on general web corpora and
overweights chunks with high numeric/lexical overlap to the query regardless of company or year -
exactly the noise financial table retrieval produces. Filter + rerank is the fix; rerank alone
is not.

### 3. Chunking strategy: null result

Tested section-aware chunking (respects SEC Item 1/1A/7/8 boundaries) against recursive
chunking (1000-token, 150-overlap) across all 4 retrieval configs and 59 answerable questions.
Section-aware scored 3.3 points *worse* on dense retrieval and tied on every other config.
The hypothesis was that preserving section boundaries would reduce cross-section noise for
boundary-sensitive questions. The data did not support it. For this corpus at chunk\_size=1000
with 150-token overlap, recursive chunking already keeps the relevant sentence intact across
most section edges. Reported as a null result.

### 4. Generation: 95.7% numerical exact-match with one traced failure

Qwen 2.5 7B achieved 95.7% exact-match on 23 numerical extraction questions (±2% tolerance for
billion/million rounding). The one failure - Goldman Sachs FY2023 net earnings, $952M predicted
vs $8,520M actual - is a table-row attribution error: retrieval found the correct chunk, but the
model selected the wrong cell from a dense income statement. This is a model-scale limitation,
not a retrieval issue.

---

## Architecture

```
30 SEC 10-K filings (10 companies x FY2022-2024)
    │
    ▼
Text extraction + metadata tagging (ticker, fiscal_year, section_id)
    │
    ├── Recursive chunker      (1000 tokens, 150 overlap)
    └── Section-aware chunker  (splits at Item boundaries)
    │
    ▼
bge-small-en-v1.5 -> ChromaDB (persistent, local)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Retrieval configs                                          │
│                                                             │
│  dense                vector similarity (cosine), top-5     │
│  hybrid               BM25 + dense, RRF fusion (k=60)      │
│  hybrid_rerank        hybrid + bge-reranker-base            │
│  hybrid_rerank_filter hybrid_rerank + NER metadata filter   │
│                       (ticker + year extracted by regex)    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Qwen 2.5 7B via Ollama (temperature 0.1)
System prompt enforces mutual exclusion: cited answer OR refusal phrase - never both.
    │
    ▼
Cited, grounded answer with [N] source attribution
```

---

## Eval Setup

**Matrix:** 2 chunking strategies x 4 retrieval configs x 65 questions = **520 retrieval runs**  
**Generation:** headline combo only (section\_aware x hybrid\_rerank\_filter), 65 questions

Ground truth hand-authored and verified against raw 10-K text:

| Question type | n (answerable) | Tests |
|---|---|---|
| single\_doc\_lookup | 10 | Headline figures (revenue, net income) |
| numerical\_extraction | 12 | Segment/product-line figures |
| cross\_company\_comparison | 10 | Multi-company comparisons |
| time\_series | 8 | Multi-year trends |
| narrative\_risk | 7 | Qualitative risk-factor content |
| boundary\_sensitive | 12 | Answers near section boundaries |
| *unanswerable* | *6* | *Out-of-corpus questions (refusal test)* |

Recall criterion: a chunk is relevant if and only if its `section_id`, `ticker`, and `fiscal_year`
all match the ground truth. A chunk from the right company in the wrong year does not count.

---

## Results Summary

| Config | Recall@5 | Recall@10 | Lat p50 |
|---|---|---|---|
| dense | 84.7% | 89.8% | 320 ms |
| hybrid | 83.1% | 93.2% | 398 ms |
| hybrid\_rerank | 79.7% | 94.9% | 402 ms |
| **hybrid\_rerank\_filter** | **94.9%** | **100.0%** | **159 ms** |

Numbers are for sec\_recursive. sec\_section\_aware matches on all configs except dense (81.4%).

Generation (sec\_section\_aware x hybrid\_rerank\_filter, v2 prompt):

| Metric | Value |
|---|---|
| Numerical exact-match | 22/23 (95.7%) |
| Refusal accuracy (unanswerable) | 5/6 |
| False refusal rate | 2/59 (3.4%) |
| Append violations | 3/65 (4.6%) |

---

## Limitations

**4.6% append-violation rate.** 3 of 65 responses append the refusal phrase to an otherwise
cited answer - a prompt-following failure in Qwen 2.5 7B. The v1 prompt produced 36.9%;
rewriting the rules as a mutually exclusive gate reduced it to 4.6%. A two-line post-processing
strip (remove the trailing phrase when `[N]` citations are present) would close the remainder.
Not applied here to keep generation outputs unmodified.

**Table-row attribution failures.** Dense financial tables (30+ numeric cells per page)
cause row-selection errors in 7B-scale models even when the correct chunk is retrieved.
The Q20 case is documented; the failure rate is hard to quantify without a larger table-specific
test set.

**Reranker domain mismatch.** bge-reranker-base's general-domain training causes a −5 point
recall@5 regression without the metadata filter. A domain-adapted or larger reranker
is the most actionable follow-on improvement on retrieval quality.

---

## What I'd Do Differently

**Vary chunk size as a third experimental dimension.** This project tested chunking strategy
at a fixed size of 1000 tokens with 150-token overlap. Section-aware chunking may only add
measurable value at smaller chunk sizes (300-500 tokens), where a naive split is more likely
to cut across a section boundary. A {300, 500, 1000} x {recursive, section-aware} grid would
have produced a cleaner answer to "does structural chunking help and at what granularity?" -
and is the first experiment I'd run if extending this work. A domain-adapted reranker and a
70B model for generation are the other two highest-leverage improvements on the current results.

---

## Running

**Requirements:** Python 3.10+, [Ollama](https://ollama.com) installed and running, ~4 GB disk.

### 1. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
ollama pull qwen2.5:7b
```

### 2. Download filings

Fetches 30 10-K filings from SEC EDGAR (AAPL, MSFT, NVDA, META, GOOGL, AMZN, JPM, GS, WMT, TSLA
x FY2022-2024). Edit the ticker/year lists at the top of the file to change the corpus.

```bash
python src/download_filings.py
```

### 3. Build the index

Extracts text, chunks into two strategies (recursive + section-aware), embeds with
bge-small-en-v1.5, and loads both collections into ChromaDB. Takes ~10 min on first run;
subsequent runs skip recomputation if the embedding cache exists.

```bash
python src/embed.py
```

### 4. Query the system

```python
from src.retrieve import retrieve
from src.generate import generate_answer

query  = "What was Apple's revenue in FY2023?"
chunks = retrieve(query, collection="sec_section_aware", config="hybrid_rerank_filter", top_k=5)
result = generate_answer(query, chunks)

print(result["answer"])
# -> "Apple reported net sales of $383.3 billion in fiscal year 2023 [1]."
```

**Collection options:** `sec_recursive`, `sec_section_aware`  
**Config options:** `dense`, `hybrid`, `hybrid_rerank`, `hybrid_rerank_filter` (recommended)

---

## Corpus

AAPL, MSFT, NVDA, META, GOOGL, AMZN, JPM, GS, WMT, TSLA x FY2022-2024 (30 filings total).

Two filings are structurally incomplete: MSFT lacks Item 7 (MD&A) across all three years; JPM
FY2024 Item 7 is absent. Ground truth questions for those filings source answers from Item 8.

---

## Project Structure

```
├── src/
│   ├── download_filings.py    # Fetches 10-Ks from SEC EDGAR
│   ├── extract.py             # Text extraction + section tagging
│   ├── chunk.py               # Recursive and section-aware chunkers
│   ├── embed.py               # Embedding + ChromaDB ingestion
│   ├── retrieve.py            # 4 retrieval configs, RRF fusion, rerank cache
│   └── generate.py            # Cited answer generation (Qwen 2.5 7B)
├── data/
│   ├── raw/manifest.csv       # Filing index (ticker, year, accession number)
│   ├── chunks/                # JSONL chunk files (generated by embed.py)
│   └── chroma/                # ChromaDB persistent storage (generated by embed.py)
└── requirements.txt
```
