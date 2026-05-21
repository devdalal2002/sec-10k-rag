# SEC 10-K RAG System

A retrieval-augmented Q&A system over 5 SEC 10-K filings (Apple, Microsoft, Nvidia, Meta, Google). Built on a fully free, local stack — no paid APIs.

## Architecture

```
User query
    │
    ▼
[Query Rewriter]  ← Llama 3.2 3B via Ollama
    │  rewrites query for better retrieval
    ▼
[Embedder]  ← all-MiniLM-L6-v2 (sentence-transformers, CPU)
    │  encodes rewritten query → 384-dim vector
    ▼
[ChromaDB]  ← cosine similarity, top-5 chunks
    │  returns relevant passages + metadata
    ▼
[Generator]  ← Llama 3.2 3B via Ollama
    │  prompt includes chunks + citations
    ▼
Answer + source citations shown in Streamlit UI
```

## Stack

| Component | Tool | Cost |
|-----------|------|------|
| LLM | Ollama + Llama 3.2 3B | Free, local |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Free, CPU |
| Vector store | ChromaDB | Free, local |
| Text extraction | pdfplumber + BeautifulSoup | Free |
| Chunking | LangChain RecursiveCharacterTextSplitter | Free |
| Frontend | Streamlit | Free |
| Hosting | Hugging Face Spaces | Free |

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/devdalal2002/sec-10k-rag
cd sec-10k-rag

# 2. Create and activate virtualenv
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Ollama and pull the model
# Download from: https://ollama.com
ollama pull llama3.2:3b

# 5. Run the Day 1 check — ALL checks must pass
python src/day1_check.py

# 6. Download the filings
python src/download_filings.py
```

## Evaluation Results

*To be filled in after Days 9–11.*

| Metric | Baseline | + Query Rewriting | Lift |
|--------|----------|-------------------|------|
| Retrieval correctness (0–2) | — | — | — |
| Answer correctness (0–2) | — | — | — |
| Faithfulness (0–2) | — | — | — |

Full results in [eval/results.md](eval/results.md).

## Example Queries

- "What was Nvidia's data center revenue in FY2024?"
- "What cybersecurity risks does Microsoft disclose in its 10-K?"
- "Compare R&D spending as a percentage of revenue across all five companies."
- "What does Apple identify as its primary supply chain risk?"

## Honest Limitations

- **Model size**: Llama 3.2 3B is small. It hallucinates on complex multi-hop questions.
- **Context window**: Chunks are 500 tokens; long tables and footnotes are truncated.
- **Retrieval**: Dense retrieval alone misses exact number lookups that need precise keyword matching.
- **Cross-document reasoning**: Comparing figures across companies requires the model to synthesize multiple chunks — failure rate is high without query rewriting.
- **Filing recency**: Filings are the most recent available at project creation (FY2023/FY2024); answers go stale.

## Project Structure

```
sec-10k-rag/
├── src/
│   ├── download_filings.py   # Fetches 10-K HTML from SEC EDGAR
│   ├── day1_check.py         # Environment sanity check
│   ├── extract.py            # Text extraction + cleaning
│   ├── chunk.py              # Recursive chunking
│   ├── embed.py              # Embedding + ChromaDB ingestion
│   ├── retrieve.py           # Query rewriting + retrieval
│   └── generate.py           # LLM generation with citations
├── eval/
│   ├── ground_truth_template.md
│   ├── ground_truth.csv      # 30 hand-written Q&A pairs
│   └── results.md            # Evaluation scores
├── notebooks/
│   └── pipeline_dev.ipynb    # End-to-end notebook (Day 5)
├── app.py                    # Streamlit frontend
├── requirements.txt
├── .gitignore
└── README.md
```
