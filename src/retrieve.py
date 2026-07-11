"""
src/retrieve.py - Four retrieval configurations over SEC 10-K Chroma collections.

Configs:
  dense                - bi-encoder cosine search only
  hybrid               - BM25 + dense, fused with RRF (k=60)
  hybrid_rerank        - hybrid candidates -> CrossEncoder rerank -> top-5
  hybrid_rerank_filter - hybrid_rerank with metadata pre-filter when
                         query names a company and/or fiscal year

Entry point:
  retrieve(query, collection, config, top_k=5) -> list[dict]
"""

import hashlib
import json
import re
import string
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CHROMA_DIR, CHUNKS_DIR, EMBED_MODEL, RERANK_MODEL,
    RRF_K, CANDIDATE_POOL, EMBED_BATCH, ADD_BATCH,
)

import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

COLLECTION_TO_JSONL = {
    "sec_recursive":     CHUNKS_DIR / "recursive.jsonl",
    "sec_section_aware": CHUNKS_DIR / "section_aware.jsonl",
}

COMPANY_MAP = {
    "apple":         "AAPL",
    "microsoft":     "MSFT",
    "nvidia":        "NVDA",
    "meta":          "META",
    "facebook":      "META",
    "google":        "GOOGL",
    "alphabet":      "GOOGL",
    "amazon":        "AMZN",
    "jpmorgan":      "JPM",
    "jp morgan":     "JPM",
    "goldman sachs": "GS",
    "goldman":       "GS",
    "walmart":       "WMT",
    "tesla":         "TSLA",
}

YEAR_RE = re.compile(r"\b(20(?:22|23|24))\b")

# ---------------------------------------------------------------------------
# Module-level lazy cache - loaded once, reused across calls
# ---------------------------------------------------------------------------
_embedder: Optional[SentenceTransformer] = None
_reranker: Optional[CrossEncoder] = None
_collection_indexes: dict = {}

# Reranker score cache: keyed by "query_hash|chunk_id" -> float score
# Persisted to disk so eval re-runs skip recomputation.
_rerank_cache: dict = {}
_rerank_cache_path: Optional[Path] = None


def load_rerank_cache(path: Path) -> None:
    global _rerank_cache, _rerank_cache_path
    _rerank_cache_path = path
    if path.exists():
        with open(path, encoding="utf-8") as f:
            _rerank_cache = json.load(f)
    else:
        _rerank_cache = {}


def save_rerank_cache() -> None:
    if _rerank_cache_path is not None:
        with open(_rerank_cache_path, "w", encoding="utf-8") as f:
            json.dump(_rerank_cache, f)


def _query_hash(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()[:16]


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    return [t for t in text.split() if t]


# ---------------------------------------------------------------------------
# Entity parsing & Chroma filter building
# ---------------------------------------------------------------------------

def _parse_entities(query: str) -> tuple[list[str], list[int]]:
    """Return (tickers, years) found in query. Multi-word names checked first."""
    q_lower = query.lower()
    tickers: list[str] = []
    # Sort by name length descending to catch "goldman sachs" before "goldman"
    for name, ticker in sorted(COMPANY_MAP.items(), key=lambda x: -len(x[0])):
        if name in q_lower and ticker not in tickers:
            tickers.append(ticker)
    years = [int(y) for y in YEAR_RE.findall(query)]
    return tickers, years


def _build_where(tickers: list[str], years: list[int]) -> Optional[dict]:
    """
    Build a Chroma `where` clause from detected entities.
    Multiple tickers -> $in (not $eq) to avoid excluding valid results.
    Returns None when nothing to filter on.
    """
    conditions: list[dict] = []

    if len(tickers) == 1:
        conditions.append({"ticker": {"$eq": tickers[0]}})
    elif len(tickers) > 1:
        conditions.append({"ticker": {"$in": tickers}})

    if len(years) == 1:
        conditions.append({"fiscal_year": {"$eq": years[0]}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _matches_filter(chunk: dict, where: dict) -> bool:
    """Check a chunk's metadata against a Chroma-style where clause."""
    if "$and" in where:
        return all(_matches_filter(chunk, cond) for cond in where["$and"])
    if "$or" in where:
        return any(_matches_filter(chunk, cond) for cond in where["$or"])
    for field, cond in where.items():
        if field.startswith("$"):
            continue
        val = chunk.get(field)
        if isinstance(cond, dict):
            if "$eq" in cond and val != cond["$eq"]:
                return False
            if "$in" in cond and val not in cond["$in"]:
                return False
        elif val != cond:
            return False
    return True


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def rrf_fuse(ranked_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """
    Reciprocal Rank Fusion over multiple ranked lists of chunk_ids.

    RRF(d) = sum_i  1 / (k + rank_i(d))   where rank is 1-indexed.

    Using rank position (not raw scores) sidesteps the incompatible scales
    problem between cosine similarity and BM25 term frequency scores.
    Standard k=60 from Cormack et al. 2009.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


# ---------------------------------------------------------------------------
# Per-collection index (BM25 + Chroma, loaded once per collection name)
# ---------------------------------------------------------------------------

class _CollectionIndex:
    """Holds the BM25 index and Chroma collection for one chunk set."""

    def __init__(self, collection_name: str):
        self.name = collection_name

        jsonl_path = COLLECTION_TO_JSONL[collection_name]
        self._chunks: dict[str, dict] = {}
        self._id_order: list[str] = []
        corpus_tokens: list[list[str]] = []

        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                self._chunks[chunk["chunk_id"]] = chunk
                self._id_order.append(chunk["chunk_id"])
                corpus_tokens.append(_tokenize(chunk["text"]))

        self.bm25 = BM25Okapi(corpus_tokens)
        self.chroma = self._load_or_build_chroma(collection_name)

    def _load_or_build_chroma(self, collection_name: str):
        """
        Load the persistent Chroma collection built by embed.py. Falls back to
        building an ephemeral, in-memory collection from the chunk JSONL when
        no persistent store is present (e.g. a fresh clone / hosted deploy
        that ships chunks but not the multi-hundred-MB Chroma directory).
        """
        try:
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            return client.get_collection(collection_name)
        except Exception:
            pass

        embedder = _get_embedder()
        client = chromadb.EphemeralClient()
        try:
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            # Leftover from an earlier build attempt that failed partway through
            # (e.g. an OOM mid-embed on a memory-constrained host) - the partial
            # collection can't be trusted, so drop it and rebuild clean.
            client.delete_collection(collection_name)
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        ids = self._id_order
        texts = [self._chunks[cid]["text"] for cid in ids]
        metadatas = [
            {
                "ticker":                self._chunks[cid]["ticker"],
                "fiscal_year":           self._chunks[cid]["fiscal_year"],
                "filing_date":           self._chunks[cid]["filing_date"],
                "strategy":              self._chunks[cid]["strategy"],
                "section_id":            self._chunks[cid]["section_id"],
                "section_title":         self._chunks[cid]["section_title"],
                "chunk_id":              self._chunks[cid]["chunk_id"],
                "char_count":            self._chunks[cid]["char_count"],
                "chunk_index_in_filing": self._chunks[cid]["chunk_index_in_filing"],
            }
            for cid in ids
        ]
        embeddings = embedder.encode(
            texts, batch_size=EMBED_BATCH, normalize_embeddings=True,
            show_progress_bar=False,
        )

        for i in range(0, len(ids), ADD_BATCH):
            collection.add(
                ids=ids[i:i + ADD_BATCH],
                embeddings=embeddings[i:i + ADD_BATCH].tolist(),
                documents=texts[i:i + ADD_BATCH],
                metadatas=metadatas[i:i + ADD_BATCH],
            )
        return collection

    # ------------------------------------------------------------------
    # Dense search via Chroma
    # ------------------------------------------------------------------

    def dense(
        self,
        query_vec: list[float],
        n: int,
        where: Optional[dict] = None,
    ) -> list[dict]:
        kwargs: dict = {
            "query_embeddings": [query_vec],
            "n_results": n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where
        res = self.chroma.query(**kwargs)
        results = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            results.append({**meta, "text": doc, "score": 1.0 - dist})
        return results

    # ------------------------------------------------------------------
    # BM25 search (returns chunk_ids in ranked order)
    # ------------------------------------------------------------------

    def bm25_ranked(
        self,
        query_tokens: list[str],
        n: int,
        where: Optional[dict] = None,
    ) -> list[str]:
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1]
        ranked_ids: list[str] = []
        for idx in ranked_indices:
            cid = self._id_order[idx]
            if where is not None and not _matches_filter(self._chunks[cid], where):
                continue
            ranked_ids.append(cid)
            if len(ranked_ids) >= n:
                break
        return ranked_ids

    def get_chunk(self, chunk_id: str) -> dict:
        return self._chunks[chunk_id]


def _get_index(collection: str) -> _CollectionIndex:
    if collection not in _collection_indexes:
        _collection_indexes[collection] = _CollectionIndex(collection)
    return _collection_indexes[collection]


def get_chunk(collection: str, chunk_id: str) -> dict:
    """Fetch a single chunk by ID without running retrieval."""
    return _get_index(collection).get_chunk(chunk_id)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    collection: str,
    config: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve top_k chunks for query from collection using the named config.

    Returns list of dicts with keys:
      chunk_id, text, section_id, ticker, fiscal_year, score,
      retrieval_method, [rerank_latency_ms if reranked]
    """
    if config not in ("dense", "hybrid", "hybrid_rerank", "hybrid_rerank_filter"):
        raise ValueError(f"Unknown config: {config!r}")

    index = _get_index(collection)
    embedder = _get_embedder()

    query_vec = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    query_tokens = _tokenize(query)

    tickers, years = _parse_entities(query)
    where = _build_where(tickers, years) if config == "hybrid_rerank_filter" else None

    # ------------------------------------------------------------------
    # dense
    # ------------------------------------------------------------------
    if config == "dense":
        results = index.dense(query_vec, top_k)
        for r in results:
            r["retrieval_method"] = "dense"
        return results[:top_k]

    # ------------------------------------------------------------------
    # hybrid  (RRF fusion, no rerank, no filter)
    # ------------------------------------------------------------------
    if config == "hybrid":
        pool = max(top_k * 4, CANDIDATE_POOL)
        dense_hits = index.dense(query_vec, pool)
        bm25_ids = index.bm25_ranked(query_tokens, pool)

        dense_ranked_ids = [r["chunk_id"] for r in dense_hits]
        fused = rrf_fuse([dense_ranked_ids, bm25_ids])

        top_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:top_k]
        return [_make_result(index.get_chunk(cid), fused[cid], "hybrid_rrf")
                for cid in top_ids]

    # ------------------------------------------------------------------
    # hybrid_rerank / hybrid_rerank_filter
    # ------------------------------------------------------------------
    pool = CANDIDATE_POOL
    dense_hits = index.dense(query_vec, pool, where=where)
    bm25_ids = index.bm25_ranked(query_tokens, pool, where=where)

    dense_ranked_ids = [r["chunk_id"] for r in dense_hits]
    fused = rrf_fuse([dense_ranked_ids, bm25_ids])

    candidate_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:pool]

    reranker = _get_reranker()
    qhash = _query_hash(query)

    # Check cache first; only call reranker for uncached pairs.
    uncached_ids = [cid for cid in candidate_ids
                    if f"{qhash}|{cid}" not in _rerank_cache]
    if uncached_ids:
        pairs = [(query, index.get_chunk(cid)["text"]) for cid in uncached_ids]
        t0 = time.perf_counter()
        scores = reranker.predict(pairs)
        rerank_ms = (time.perf_counter() - t0) * 1000
        for cid, score in zip(uncached_ids, scores):
            _rerank_cache[f"{qhash}|{cid}"] = float(score)
    else:
        rerank_ms = 0.0

    rerank_scores = [_rerank_cache[f"{qhash}|{cid}"] for cid in candidate_ids]

    # Sanity-check: scores must not all be identical
    if len(set(rerank_scores)) == 1:
        raise RuntimeError(
            f"Reranker returned identical scores for all {len(candidate_ids)} candidates. "
            "Model may not have loaded correctly."
        )

    ranked = sorted(zip(candidate_ids, rerank_scores), key=lambda x: x[1], reverse=True)
    results = []
    for cid, score in ranked[:top_k]:
        r = _make_result(index.get_chunk(cid), float(score), config)
        r["rerank_latency_ms"] = rerank_ms
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _make_result(chunk: dict, score: float, method: str) -> dict:
    return {
        "chunk_id":              chunk["chunk_id"],
        "ticker":                chunk["ticker"],
        "fiscal_year":           chunk["fiscal_year"],
        "section_id":            chunk["section_id"],
        "section_title":         chunk["section_title"],
        "text":                  chunk["text"],
        "score":                 score,
        "retrieval_method":      method,
    }
