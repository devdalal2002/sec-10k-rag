"""
src/embed.py — Embed both chunk sets and load into two Chroma collections.

Collections:
  sec_recursive     <- data/chunks/recursive.jsonl
  sec_section_aware <- data/chunks/section_aware.jsonl

Persistence: data/chroma/
"""

import json
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_DIR = Path("data/chunks")
CHROMA_DIR = Path("data/chroma")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_BATCH = 64
ADD_BATCH = 2000  # max items per Chroma .add() call


def load_jsonl(path: Path) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def embed_and_index(
    name: str,
    jsonl_path: Path,
    model: SentenceTransformer,
    client: chromadb.PersistentClient,
) -> None:
    print(f"\n{'='*50}")
    print(f"Collection: {name}")

    try:
        client.delete_collection(name)
        print("  Dropped existing collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )

    chunks = load_jsonl(jsonl_path)
    print(f"  {len(chunks):,} chunks loaded")

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {
            "ticker":                c["ticker"],
            "fiscal_year":           c["fiscal_year"],
            "filing_date":           c["filing_date"],
            "strategy":              c["strategy"],
            "section_id":            c["section_id"],
            "section_title":         c["section_title"],
            "chunk_id":              c["chunk_id"],
            "char_count":            c["char_count"],
            "chunk_index_in_filing": c["chunk_index_in_filing"],
        }
        for c in chunks
    ]

    print(f"  Embedding ({MODEL_NAME}, batch={EMBED_BATCH})...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embed_secs = time.time() - t0
    print(f"  Embedding done: {embed_secs:.1f}s ({embed_secs/60:.1f} min)")

    print(f"  Indexing into Chroma (batch={ADD_BATCH})...")
    for i in range(0, len(chunks), ADD_BATCH):
        collection.add(
            ids=ids[i:i + ADD_BATCH],
            embeddings=embeddings[i:i + ADD_BATCH].tolist(),
            documents=texts[i:i + ADD_BATCH],
            metadatas=metadatas[i:i + ADD_BATCH],
        )
    print(f"  Indexed {collection.count():,} vectors")


def main():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    t_total = time.time()

    embed_and_index("sec_recursive",     CHUNKS_DIR / "recursive.jsonl",     model, client)
    embed_and_index("sec_section_aware", CHUNKS_DIR / "section_aware.jsonl", model, client)

    elapsed = time.time() - t_total
    print(f"\nTotal wall time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
