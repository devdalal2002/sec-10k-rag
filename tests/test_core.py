"""
tests/test_core.py - Unit tests for is_refusal, numerical_match, score_recall.

Run:  python tests/test_core.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval import is_refusal, numerical_match, score_recall

REFUSAL = "The provided filings do not contain this information."


class TestIsRefusal(unittest.TestCase):
    def test_pure_refusal(self):
        self.assertTrue(is_refusal(REFUSAL))

    def test_pure_refusal_lowercase(self):
        self.assertTrue(is_refusal(REFUSAL.lower()))

    def test_answer_with_citations_not_refusal(self):
        answer = "Apple reported $49,552M in net income [1]. The provided filings do not contain this information."
        self.assertFalse(is_refusal(answer))

    def test_clean_cited_answer_not_refusal(self):
        answer = "Microsoft's revenue was $211.9 billion in FY2023 [2][3]."
        self.assertFalse(is_refusal(answer))

    def test_short_denial_refusal(self):
        answer = "I cannot answer this from the provided excerpts."
        self.assertTrue(is_refusal(answer))

    def test_short_not_contain_refusal(self):
        answer = "The excerpts do not contain this information."
        self.assertTrue(is_refusal(answer))

    def test_long_answer_without_phrase_not_refusal(self):
        answer = "Tesla's revenue grew significantly. " * 10
        self.assertFalse(is_refusal(answer))

    def test_answer_with_bracket_citation_style(self):
        answer = "Net income was $8,520M [N1][N2]. The provided filings do not contain this information."
        self.assertFalse(is_refusal(answer))


class TestNumericalMatch(unittest.TestCase):
    def test_exact_millions(self):
        self.assertTrue(numerical_match("49552", "Apple reported $49,552 million [1]."))

    def test_billion_rounding(self):
        self.assertTrue(numerical_match("49552", "Apple reported $49.6 billion [1]."))

    def test_exact_billion(self):
        self.assertTrue(numerical_match("211900", "Revenue was $211.9 billion [2]."))

    def test_wrong_value(self):
        self.assertFalse(numerical_match("49552", "Revenue was $952 million [1]."))

    def test_empty_expected(self):
        self.assertFalse(numerical_match("", "Revenue was $49 billion [1]."))

    def test_nonnumeric_expected(self):
        self.assertFalse(numerical_match("N/A", "Revenue was $49 billion [1]."))

    def test_within_2pct_tolerance(self):
        # 60900 vs 60922: (22/60922) = 0.036% < 2%
        self.assertTrue(numerical_match("60922", "Revenue was $60.9 billion [1]."))

    def test_outside_tolerance(self):
        # 10000 vs 10300 = 3% > 2%
        self.assertFalse(numerical_match("10000", "Revenue was $10.3 billion [1]."))

    def test_trillion(self):
        self.assertTrue(numerical_match("1000000", "Total assets were $1.0 trillion [3]."))


class TestScoreRecall(unittest.TestCase):
    def _gt_row(self):
        return {
            "relevant_section_ids": "item_7;item_7a",
            "relevant_tickers":     "AAPL",
            "relevant_years":       "2023",
        }

    def _chunk(self, section_id="item_7", ticker="AAPL", fiscal_year=2023):
        return {
            "chunk_id":   f"{ticker}_{fiscal_year}_{section_id}_0",
            "section_id": section_id,
            "ticker":     ticker,
            "fiscal_year": fiscal_year,
            "text":       "...",
        }

    def test_hit_in_top_k(self):
        chunks = [self._chunk()]
        self.assertTrue(score_recall(chunks, self._gt_row(), k=5))

    def test_miss_wrong_ticker(self):
        chunks = [self._chunk(ticker="MSFT")]
        self.assertFalse(score_recall(chunks, self._gt_row(), k=5))

    def test_miss_wrong_year(self):
        chunks = [self._chunk(fiscal_year=2022)]
        self.assertFalse(score_recall(chunks, self._gt_row(), k=5))

    def test_miss_wrong_section(self):
        chunks = [self._chunk(section_id="item_1")]
        self.assertFalse(score_recall(chunks, self._gt_row(), k=5))

    def test_hit_on_secondary_section(self):
        chunks = [self._chunk(section_id="item_7a")]
        self.assertTrue(score_recall(chunks, self._gt_row(), k=5))

    def test_hit_outside_k_window(self):
        filler = [self._chunk(section_id="item_1") for _ in range(5)]
        hit    = self._chunk()
        chunks = filler + [hit]
        self.assertFalse(score_recall(chunks, self._gt_row(), k=5))
        self.assertTrue(score_recall(chunks, self._gt_row(), k=6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
