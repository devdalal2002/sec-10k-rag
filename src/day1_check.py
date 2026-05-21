"""
Day 1 sanity check — run this before anything else.
All three checks must pass before you proceed to Day 2.

Usage:
    python src/day1_check.py
"""

import sys


def check_python_version():
    print("Check 1: Python version")
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 10:
        print(f"  [PASS] Python {major}.{minor}")
        return True
    else:
        print(f"  [FAIL] Python {major}.{minor} — need 3.10+")
        print("         Download from https://python.org")
        return False


def check_dependencies():
    print("\nCheck 2: Python dependencies")
    deps = [
        ("pdfplumber", "pdfplumber"),
        ("langchain", "langchain"),
        ("sentence_transformers", "sentence-transformers"),
        ("chromadb", "chromadb"),
        ("ollama", "ollama"),
        ("streamlit", "streamlit"),
        ("requests", "requests"),
        ("bs4", "beautifulsoup4"),
    ]
    all_ok = True
    for module, pkg in deps:
        try:
            __import__(module)
            print(f"  [ok] {pkg}")
        except ImportError:
            print(f"  [MISSING] {pkg}  ->  pip install {pkg}")
            all_ok = False
    if all_ok:
        print("  [PASS] All dependencies installed")
    else:
        print("  [FAIL] Run: pip install -r requirements.txt")
    return all_ok


def check_embeddings():
    print("\nCheck 3: Embedding model (sentence-transformers)")
    try:
        from sentence_transformers import SentenceTransformer
        print("  Loading all-MiniLM-L6-v2 (downloads ~90 MB on first run)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vec = model.encode(["SEC 10-K filing test sentence"])
        assert len(vec[0]) == 384, f"Unexpected dimension: {len(vec[0])}"
        print(f"  [PASS] Embedding produced vector of dimension {len(vec[0])}")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def check_ollama():
    print("\nCheck 4: Ollama + Llama 3.2 3B")
    try:
        import ollama
        resp = ollama.chat(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": "Reply with just the word PASS"}],
        )
        reply = resp["message"]["content"].strip()
        print(f"  Model replied: '{reply}'")
        print("  [PASS] Ollama is running and llama3.2:3b is available")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        print("  Fix steps:")
        print("    1. Install Ollama: https://ollama.com")
        print("    2. Start it:       ollama serve  (or open the Ollama app)")
        print("    3. Pull the model: ollama pull llama3.2:3b")
        print("    4. Re-run this script")
        return False


def main():
    print("=" * 55)
    print("  SEC 10-K RAG — Day 1 Environment Check")
    print("=" * 55)

    results = [
        check_python_version(),
        check_dependencies(),
        check_embeddings(),
        check_ollama(),
    ]

    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 55)
    print(f"  Result: {passed}/{total} checks passed")
    print("=" * 55)

    if passed == total:
        print("\nAll checks passed. You are ready for Day 2.")
        print("Next: python src/download_filings.py")
    else:
        print("\nFix the failing checks above before continuing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
