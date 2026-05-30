"""
tests/test_numerical_match.py - Unit tests for numerical_match().

Run: pytest tests/test_numerical_match.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.run_eval import numerical_match


def test_exact_millions():
    assert numerical_match("49552", "Apple reported $49,552 million [1].")

def test_billion_rounding():
    assert numerical_match("49552", "Apple reported $49.6 billion [1].")

def test_exact_billion():
    assert numerical_match("211900", "Revenue was $211.9 billion [2].")

def test_within_2pct_tolerance():
    # (60922 - 60900) / 60922 = 0.036% < 2%
    assert numerical_match("60922", "Revenue was $60.9 billion [1].")

def test_trillion():
    assert numerical_match("1000000", "Total assets were $1.0 trillion [3].")

def test_wrong_value():
    assert not numerical_match("49552", "Revenue was $952 million [1].")

def test_outside_tolerance():
    # (10300 - 10000) / 10000 = 3% > 2%
    assert not numerical_match("10000", "Revenue was $10.3 billion [1].")

def test_empty_expected():
    assert not numerical_match("", "Revenue was $49 billion [1].")

def test_nonnumeric_expected():
    assert not numerical_match("N/A", "Revenue was $49 billion [1].")
