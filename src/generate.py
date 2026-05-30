"""
src/generate.py - Generation layer: retrieved chunks -> grounded, cited answer.

Uses Ollama with qwen2.5:7b (better numerical grounding than llama3.2:3b).
Temperature 0.1 - extraction task, not creative writing.
"""

import re
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import LLM_MODEL as DEFAULT_MODEL, LLM_TEMPERATURE as TEMPERATURE

import ollama

SYSTEM_PROMPT = """\
You are a financial analyst assistant. Answer questions using ONLY the SEC 10-K filing excerpts provided below.

Rules:
1. Cite the source of every factual claim using [N] notation, where N is the excerpt number.
2. If multiple excerpts support a claim, cite all of them: [1][3].
3. Answer only from the provided excerpts. Do not use any prior knowledge.
4. Be precise with numbers - quote figures directly from the excerpts rather than paraphrasing them.
5. Do not speculate or infer figures that are not stated explicitly.
6. Choose EXACTLY ONE of the two response forms below - never both:
   FORM A (use when ANY excerpt contains relevant information, even partial):
     Write an answer with [N] citations. If only part of the question can be answered,
     state what the excerpts support (with citations) and note what is missing in your
     own words. Do NOT append the refusal phrase from Form B.
   FORM B (use ONLY when NO excerpt contains any relevant information at all):
     Respond with exactly this single sentence and nothing else:
     "The provided filings do not contain this information."
     Do NOT use Form B if your response already includes any [N] citations.\
"""


def _build_context(chunks: list[dict]) -> str:
    """Format numbered excerpt block for the user message."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        ticker = chunk.get("ticker", "?")
        year   = chunk.get("fiscal_year", "?")
        sec    = chunk.get("section_id", "?")
        text   = chunk.get("text", "").strip()
        lines.append(f"[{i}] ({ticker} FY{year}, {sec}):\n{text}")
    return "\n\n".join(lines)


def _parse_citations(answer: str, num_chunks: int) -> list[int]:
    """
    Extract [N] citation indices from the model answer.
    Returns sorted list of 1-based indices that are valid (within range).
    Invalid indices (hallucinated out-of-range numbers) are silently dropped.
    """
    raw = [int(m) for m in re.findall(r"\[(\d+)\]", answer)]
    valid = sorted(set(n for n in raw if 1 <= n <= num_chunks))
    return valid


def generate_answer(
    query: str,
    chunks: list[dict],
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Generate a grounded, cited answer from retrieved chunks.

    Returns:
      answer          - model response text
      citations       - list of {chunk_id, ticker, fiscal_year, section_id}
                        for each chunk actually cited (validated against range)
      hallucinated_citations - indices the model cited that were out of range
      model           - model name used
      num_chunks_used - how many chunks were passed in
      generation_ms   - wall-clock time for the Ollama call
    """
    context = _build_context(chunks)
    user_message = f"Question: {query}\n\nExcerpts:\n{context}\n\nAnswer:"

    t0 = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        options={"temperature": TEMPERATURE},
    )
    generation_ms = (time.perf_counter() - t0) * 1000

    answer = response["message"]["content"].strip()

    valid_indices = _parse_citations(answer, len(chunks))
    raw_indices   = sorted(set(
        int(m) for m in re.findall(r"\[(\d+)\]", answer)
    ))
    hallucinated  = [n for n in raw_indices if n < 1 or n > len(chunks)]

    citations = [
        {
            "chunk_id":    chunks[i - 1]["chunk_id"],
            "ticker":      chunks[i - 1].get("ticker", "?"),
            "fiscal_year": chunks[i - 1].get("fiscal_year", "?"),
            "section_id":  chunks[i - 1].get("section_id", "?"),
        }
        for i in valid_indices
    ]

    return {
        "answer":                 answer,
        "citations":              citations,
        "hallucinated_citations": hallucinated,
        "model":                  model,
        "num_chunks_used":        len(chunks),
        "generation_ms":          generation_ms,
    }
