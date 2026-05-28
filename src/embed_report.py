"""
src/embed_report.py — Collection stats and sanity retrieval query.
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path("data/chroma")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
SANITY_QUERY = "What were Nvidia's revenue figures?"


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 / 1024


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    model = SentenceTransformer(MODEL_NAME)

    # --- Collection table ---
    print(f"\n{'Collection':<26} {'Count':>8}  {'Dim':>5}")
    print("-" * 45)
    for name in ["sec_recursive", "sec_section_aware"]:
        coll = client.get_collection(name)
        count = coll.count()
        sample = coll.get(limit=1, include=["embeddings"])
        dim = len(sample["embeddings"][0]) if sample["embeddings"] is not None and len(sample["embeddings"]) > 0 else "?"
        print(f"{name:<26} {count:>8,}  {dim:>5}")

    total_mb = dir_size_mb(CHROMA_DIR)
    print(f"\nTotal Chroma dir: {total_mb:.1f} MB")

    # --- Sanity query ---
    query_vec = model.encode([SANITY_QUERY], normalize_embeddings=True)[0].tolist()

    print(f"\nSanity query: \"{SANITY_QUERY}\"")
    print("=" * 70)

    for name in ["sec_recursive", "sec_section_aware"]:
        coll = client.get_collection(name)
        res = coll.query(
            query_embeddings=[query_vec],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )
        print(f"\n{name}:")
        for i, (doc, meta, dist) in enumerate(zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        )):
            # Chroma cosine space: distance = 1 - similarity
            score = 1.0 - dist
            print(f"  [{i+1}] {meta['chunk_id']}")
            print(f"       section: {meta['section_id']}  ticker: {meta['ticker']}  year: {meta['fiscal_year']}  score: {score:.4f}")
            print(f"       text: {doc[:200]}")
            print()


if __name__ == "__main__":
    main()
