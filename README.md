# SEC 10-K RAG System

Retrieval-augmented generation over 30 SEC 10-K filings (10 companies x FY2022-2024, ~22K chunks),
evaluated on a hand-verified 65-question benchmark across 8 retrieval configurations.

**94.9% recall@5, 100% recall@10** with named-entity metadata filtering. Full numbers: [eval/results.md](eval/results.md)

**Four findings:**
- Metadata filtering (ticker + year extracted from query) is the dominant lever: +10-15 pts recall@5
- Reranking alone *hurts* recall@5 by 5 pts without filtering — cross-encoder domain mismatch
- Section-aware chunking: null result vs recursive at chunk_size=1000 (tied on all 4 configs)
- 95.7% numerical exact-match (+-2% tolerance); one traced failure documented in results.md

## Quickstart

```bash
git clone https://github.com/devdalal2002/sec-10k-rag
cd sec-10k-rag
./scripts/setup.sh                          # venv + pip install + ollama pull
python src/download_filings.py              # fetch 30 filings from SEC EDGAR
python src/embed.py                         # extract -> chunk -> embed (~10 min)
python eval/run_eval.py --sample 10        # quick smoke-test (10 questions)
python eval/run_eval.py                     # full run: 520 retrieval + 65 generation
```

To query interactively after building the index:

```python
from src.retrieve import retrieve
from src.generate import generate_answer

query  = "What was Apple's revenue in FY2023?"
chunks = retrieve(query, collection="sec_section_aware", config="hybrid_rerank_filter")
print(generate_answer(query, chunks)["answer"])
# -> "Apple reported net sales of $383.3 billion in fiscal year 2023 [1]."
```

## Streamlit UI

A minimal chat UI (`src/app.py`) wraps `retrieve()` + `generate_answer()`, fixed to the
headline eval config (section-aware chunking, hybrid + rerank + metadata filter, top-5 -
94.9% recall@5 / 100% recall@10 per [eval/results.md](eval/results.md)). Click the info
button for corpus/backend details, or use the example-question shortcuts to try it.

```bash
streamlit run src/app.py
```

By default it uses the local Ollama backend and the persistent Chroma index built by
`embed.py`. It falls back to an in-memory index built from `data/chunks/*.jsonl` when
`data/chroma/` isn't present (e.g. a fresh clone) - see below for hosting it live.

### Deploying to Streamlit Community Cloud

Ollama can't run on Streamlit Cloud, so the hosted app needs a different generation
backend and can't ship the ~300 MB local Chroma index. Both are handled automatically:

- **Generation**: set a `GROQ_API_KEY` secret (free tier, get one at
  [console.groq.com/keys](https://console.groq.com/keys)) in the app's Secrets settings.
  `src/app.py` detects the secret and switches from Ollama/qwen2.5:7b to
  Groq/llama-3.3-70b-versatile automatically - see `.streamlit/secrets.toml.example`.
- **Retrieval index**: `data/chunks/*.jsonl` (the chunked filing text, ~41 MB) and
  `data/embeddings/sec_section_aware.npy` (precomputed bge-small embeddings, ~31 MB) are both
  tracked in git so the app can load the array directly into an ephemeral in-memory Chroma
  collection on first query, instead of re-encoding ~21K chunks at runtime - the latter took
  several minutes or more on Streamlit Cloud's shared CPU before this was added. The reranker
  is also skipped on this backend (see below) since `bge-reranker-base` alone is ~1.1 GB,
  more than the free tier's RAM comfortably allows alongside everything else.
- **Dependencies**: Streamlit Cloud installs whichever `requirements.txt` is closest to the
  entrypoint, so `src/requirements.txt` (a lean subset: streamlit, chromadb,
  sentence-transformers, rank-bm25, ollama, groq) is used for the hosted deploy instead of
  the root `requirements.txt`. The offline ingestion scripts (`extract.py`,
  `download_filings.py`, `chunk.py`) need `pdfplumber`/`beautifulsoup4`/`lxml`/`langchain*`,
  but `src/app.py` never imports them - and some of those pins lack prebuilt wheels on
  Streamlit Cloud's Python version, so keeping them out of the hosted install avoids a
  build failure.

To deploy: push this repo to GitHub, create an app at
[share.streamlit.io](https://share.streamlit.io) pointing at `src/app.py`, and add the
`GROQ_API_KEY` secret.

---

## Four Findings

### 1. Metadata filtering is the dominant lever (+10-15 pts recall@5)

Named-entity filtering - extracting ticker and fiscal year from the query and narrowing the
candidate pool before reranking - lifted recall@5 from ~84% to 94.9% and recall@10 to 100%.
The mechanism: filtering reduces the retrieval problem from "find the right chunk in 22K" to
"find the right chunk among ~300 same-company/year chunks," eliminating the main source of
false positives before any ranking happens.

### 2. Reranking alone decreases recall@5 (-5 points)

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

Qwen 2.5 7B achieved 95.7% exact-match on 23 numerical extraction questions (+-2% tolerance for
billion/million rounding). The one failure - Goldman Sachs FY2023 net earnings, $952M predicted
vs $8,520M actual - is a table-row attribution error: retrieval found the correct chunk, but the
model selected the wrong cell from a dense income statement. This is a model-scale limitation,
not a retrieval issue. Documented in [eval/results.md](eval/results.md).

---

## Architecture

```
30 SEC 10-K filings (10 companies x FY2022-2024)
    |
    v
Text extraction + metadata tagging (ticker, fiscal_year, section_id)
    |
    +-- Recursive chunker      (1000 tokens, 150 overlap)
    +-- Section-aware chunker  (splits at Item boundaries)
    |
    v
bge-small-en-v1.5 -> ChromaDB (persistent, local)
    |
    v
+-------------------------------------------------------------+
|  Retrieval configs                                          |
|                                                             |
|  dense                vector similarity (cosine), top-5    |
|  hybrid               BM25 + dense, RRF fusion (k=60)     |
|  hybrid_rerank        hybrid + bge-reranker-base           |
|  hybrid_rerank_filter hybrid_rerank + NER metadata filter  |
|                       (ticker + year extracted by regex)   |
+-------------------------------------------------------------+
    |
    v
Qwen 2.5 7B via Ollama (temperature 0.1)
System prompt enforces mutual exclusion: cited answer OR refusal phrase - never both.
    |
    v
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
The Q20 case is documented in [eval/results.md](eval/results.md); the failure rate is hard to
quantify without a larger table-specific test set.

**Reranker domain mismatch.** bge-reranker-base's general-domain training causes a -5 point
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

## Corpus

AAPL, MSFT, NVDA, META, GOOGL, AMZN, JPM, GS, WMT, TSLA x FY2022-2024 (30 filings total).

Two filings are structurally incomplete: MSFT lacks Item 7 (MD&A) across all three years; JPM
FY2024 Item 7 is absent. Ground truth questions for those filings source answers from Item 8.

---

## Project Structure

```
sec-10k-rag/
├── src/
│   ├── config.py              # tunable params: model names, chunk size, paths
│   ├── download_filings.py    # fetch 10-Ks from SEC EDGAR
│   ├── extract.py             # HTML -> structured JSON with section tags
│   ├── chunk.py               # recursive + section-aware chunkers
│   ├── embed.py               # embed chunks -> ChromaDB
│   ├── retrieve.py            # 4 retrieval configs, RRF fusion, rerank cache
│   ├── generate.py            # cited answer generation (Ollama or Groq)
│   └── app.py                 # Streamlit chat UI
├── eval/
│   ├── ground_truth.csv       # 65 hand-verified Q&A pairs (tracked)
│   ├── raw_results.jsonl      # per-(question, collection, config) outcomes (tracked)
│   ├── results.md             # summary tables and findings
│   └── run_eval.py            # harness: 520 retrieval + 65 generation runs
├── tests/
│   ├── test_chunk.py          # section-aware chunks never cross section boundaries
│   ├── test_retrieve.py       # metadata filter entity parsing and matching
│   └── test_numerical_match.py
├── scripts/
│   ├── setup.sh               # venv + pip + ollama pull
│   └── run_full_pipeline.sh   # download -> index -> eval in one command
├── data/
│   ├── raw/manifest.csv       # filing index (tracked)
│   ├── processed/             # generated by extract.py (gitignored)
│   ├── chunks/                # generated by chunk.py (gitignored)
│   ├── chroma/                # ChromaDB store (gitignored)
│   └── cache/                 # rerank score cache (gitignored)
├── README.md
├── requirements.txt
└── LICENSE
```
