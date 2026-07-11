"""
src/config.py - Single source of truth for tunable parameters and paths.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
COMPANIES = ["AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "JPM", "GS", "WMT", "TSLA"]
YEARS     = [2022, 2023, 2024]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR      = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR    = DATA_DIR / "chunks"
CHROMA_DIR    = DATA_DIR / "chroma"
CACHE_DIR     = DATA_DIR / "cache"
RERANK_CACHE  = CACHE_DIR / "rerank_cache.json"
# Precomputed embeddings tracked in git (unlike data/chroma/) so hosted deploys
# without a persistent Chroma store can load them instead of re-encoding the
# full corpus at runtime - see retrieve.py's ephemeral-index fallback.
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 150
MIN_SECTION_CHARS = 200
MIN_CHUNK_CHARS   = 100

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_BATCH = 64
ADD_BATCH   = 100   # Chroma safe batch ceiling (~166 max)

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
RERANK_MODEL   = "BAAI/bge-reranker-base"
RRF_K          = 60   # Cormack et al. 2009 standard
CANDIDATE_POOL = 20   # candidates fed to the cross-encoder
DEFAULT_TOP_K  = 5

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
# "ollama" (default, local) or "groq" (hosted deploys without a local LLM server).
# app.py sets LLM_BACKEND=groq automatically when a GROQ_API_KEY secret is present.
LLM_BACKEND     = os.environ.get("LLM_BACKEND", "ollama")
LLM_MODEL       = "qwen2.5:7b"
GROQ_MODEL      = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = 0.1
