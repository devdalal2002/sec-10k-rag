"""
Day 1 sanity check. Run this AFTER you set up the environment.

If all three checks pass, you are ready to start Day 2.
"""
import sys

def check_imports():
    try:
        import langchain
        import chromadb
        import sentence_transformers
        import pdfplumber
        import streamlit
        print("[ok] All Python deps importable")
        return True
    except ImportError as e:
        print(f"[fail] Missing dep: {e}")
        return False

def check_ollama():
    import requests
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if any("llama3.2" in m for m in models):
            print(f"[ok] Ollama running with llama3.2: {models}")
            return True
        else:
            print(f"[fail] Ollama running but llama3.2 not found. Run: ollama pull llama3.2:3b")
            return False
    except Exception as e:
        print(f"[fail] Ollama not running. Start it and run: ollama pull llama3.2:3b. Error: {e}")
        return False

def check_embeddings():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        emb = model.encode(["test sentence"])
        assert emb.shape == (1, 384), f"Unexpected shape: {emb.shape}"
        print(f"[ok] Embeddings working, shape {emb.shape}")
        return True
    except Exception as e:
        print(f"[fail] Embeddings error: {e}")
        return False

if __name__ == "__main__":
    results = [check_imports(), check_ollama(), check_embeddings()]
    if all(results):
        print("\nAll checks passed. Proceed to Day 2.")
        sys.exit(0)
    else:
        print("\nFix failures before proceeding.")
        sys.exit(1)
