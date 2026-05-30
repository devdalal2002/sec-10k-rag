"""
tests/test_retrieve.py - Tests for metadata filtering and entity parsing.

Run: pytest tests/test_retrieve.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieve import _parse_entities, _build_where, _matches_filter


class TestParseEntities:
    def test_single_company(self):
        tickers, years = _parse_entities("What was Apple's revenue in 2023?")
        assert tickers == ["AAPL"]
        assert years == [2023]

    def test_multi_word_company(self):
        tickers, years = _parse_entities("Goldman Sachs net income 2022")
        assert "GS" in tickers
        assert years == [2022]

    def test_no_entities(self):
        tickers, years = _parse_entities("What is the largest risk factor?")
        assert tickers == []
        assert years == []

    def test_multiple_companies(self):
        tickers, years = _parse_entities("Compare Apple and Microsoft revenue in 2023")
        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert 2023 in years

    def test_facebook_alias(self):
        tickers, years = _parse_entities("Facebook revenue 2023")
        assert "META" in tickers

    def test_year_out_of_corpus_ignored(self):
        _, years = _parse_entities("Apple revenue in 2019")
        assert 2019 not in years   # YEAR_RE only matches 2022-2024


class TestBuildWhere:
    def test_single_ticker_single_year(self):
        where = _build_where(["AAPL"], [2023])
        assert where == {"$and": [{"ticker": {"$eq": "AAPL"}}, {"fiscal_year": {"$eq": 2023}}]}

    def test_single_ticker_no_year(self):
        where = _build_where(["MSFT"], [])
        assert where == {"ticker": {"$eq": "MSFT"}}

    def test_multiple_tickers(self):
        where = _build_where(["AAPL", "MSFT"], [2023])
        assert "$and" in where
        conds = where["$and"]
        ticker_cond = next(c for c in conds if "ticker" in c)
        assert "$in" in ticker_cond["ticker"]
        assert set(ticker_cond["ticker"]["$in"]) == {"AAPL", "MSFT"}

    def test_no_entities_returns_none(self):
        assert _build_where([], []) is None

    def test_multiple_years_no_filter(self):
        # Only single year triggers a year filter
        where = _build_where(["AAPL"], [2022, 2023])
        assert where == {"ticker": {"$eq": "AAPL"}}


class TestMatchesFilter:
    def _chunk(self, ticker="AAPL", fiscal_year=2023):
        return {"ticker": ticker, "fiscal_year": fiscal_year, "section_id": "item_7"}

    def test_eq_match(self):
        where = {"ticker": {"$eq": "AAPL"}}
        assert _matches_filter(self._chunk("AAPL"), where)
        assert not _matches_filter(self._chunk("MSFT"), where)

    def test_in_match(self):
        where = {"ticker": {"$in": ["AAPL", "MSFT"]}}
        assert _matches_filter(self._chunk("AAPL"), where)
        assert _matches_filter(self._chunk("MSFT"), where)
        assert not _matches_filter(self._chunk("NVDA"), where)

    def test_and_clause(self):
        where = {"$and": [{"ticker": {"$eq": "AAPL"}}, {"fiscal_year": {"$eq": 2023}}]}
        assert _matches_filter(self._chunk("AAPL", 2023), where)
        assert not _matches_filter(self._chunk("AAPL", 2022), where)
        assert not _matches_filter(self._chunk("MSFT", 2023), where)
