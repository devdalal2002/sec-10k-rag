"""
tests/test_chunk.py - Tests for chunking strategies.

Run: pytest tests/test_chunk.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chunk import chunk_recursive, chunk_section_aware


def _make_filing(sections: list[dict]) -> dict:
    return {
        "ticker": "TEST",
        "fiscal_year": 2023,
        "filing_date": "2024-01-01",
        "sections": sections,
    }


def _make_section(section_id: str, text: str) -> dict:
    return {
        "section_id": section_id,
        "section_title": section_id.upper(),
        "text": text,
        "char_count": len(text),
    }


class TestSectionAware:
    def test_chunks_stay_within_section(self):
        """Every chunk must be a substring of its source section."""
        sec_a = _make_section("item_7", "Revenue grew significantly. " * 50)
        sec_b = _make_section("item_7a", "Risk factors include market volatility. " * 50)
        filing = _make_filing([sec_a, sec_b])

        chunks = chunk_section_aware(filing)
        assert len(chunks) > 0

        for chunk in chunks:
            if chunk["section_id"] == "item_7":
                assert chunk["text"] in sec_a["text"], "Chunk crossed into wrong section"
            elif chunk["section_id"] == "item_7a":
                assert chunk["text"] in sec_b["text"], "Chunk crossed into wrong section"

    def test_section_id_assigned_correctly(self):
        sec_a = _make_section("item_1", "Business description. " * 60)
        sec_b = _make_section("item_2", "Risk factors details. " * 60)
        filing = _make_filing([sec_a, sec_b])

        chunks = chunk_section_aware(filing)
        ids = {c["section_id"] for c in chunks}
        assert "item_1" in ids
        assert "item_2" in ids

    def test_no_cross_section_text(self):
        """Text unique to section B must never appear in a section A chunk."""
        unique_b = "UNIQUE_MARKER_FOR_SECTION_B"
        sec_a = _make_section("item_7", "Normal revenue text. " * 60)
        sec_b = _make_section("item_8", f"{unique_b} " + "Financial statements. " * 60)
        filing = _make_filing([sec_a, sec_b])

        chunks = chunk_section_aware(filing)
        a_chunks = [c for c in chunks if c["section_id"] == "item_7"]
        assert all(unique_b not in c["text"] for a_chunks in [a_chunks] for c in a_chunks)

    def test_small_sections_skipped(self):
        tiny = _make_section("item_9", "Too short.")   # char_count < MIN_SECTION_CHARS
        normal = _make_section("item_7", "Normal text content here. " * 50)
        filing = _make_filing([tiny, normal])

        chunks = chunk_section_aware(filing)
        assert all(c["section_id"] != "item_9" for c in chunks)


class TestRecursive:
    def test_produces_chunks(self):
        sec = _make_section("item_7", "Revenue grew significantly. " * 100)
        filing = _make_filing([sec])
        chunks = chunk_recursive(filing)
        assert len(chunks) > 0

    def test_metadata_fields_present(self):
        sec = _make_section("item_7", "Revenue grew. " * 100)
        filing = _make_filing([sec])
        chunks = chunk_recursive(filing)
        for c in chunks:
            assert "chunk_id" in c
            assert "ticker" in c
            assert "fiscal_year" in c
            assert "section_id" in c
            assert "text" in c
