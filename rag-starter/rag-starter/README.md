# Financial Document RAG System

A retrieval-augmented Q&A system over SEC 10-K annual reports. Built with local open-source models (Ollama + sentence-transformers + ChromaDB) so it runs entirely free.

## Status

In development. See `eval/results.md` for current performance.

## Architecture

```
PDF 10-K filings
       v
pdfplumber text extraction
       v
Recursive chunking (500 tokens, 50 overlap)
       v
sentence-transformers embeddings (MiniLM-L6-v2)
       v
ChromaDB vector store
       v
[Query] --> Query rewriting (Llama 3.2) --> Dense retrieval (top-5)
       v
Llama 3.2 3B answer generation with citations
       v
Streamlit UI
```

## Setup

```bash
# 1. Install Ollama from https://ollama.com
ollama pull llama3.2:3b

# 2. Python deps
pip install -r requirements.txt

# 3. Download SEC filings
python src/download_filings.py

# 4. Build vector store
python src/build_index.py

# 5. Run app
streamlit run src/app.py
```

## Evaluation

30-question hand-curated eval set. Scoring: retrieval correctness, answer correctness, faithfulness. See `eval/` for details.

## Limitations

- Local LLM is weaker than GPT-4 class models, expect imperfect answers
- No multi-document reasoning across companies
- Numerical extraction from tables is unreliable

## Tech

Python, LangChain, sentence-transformers, ChromaDB, Ollama, Streamlit, pdfplumber
