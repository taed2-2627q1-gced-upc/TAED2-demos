"""Antipattern 2: shared mutable state. Tests pass or fail depending on execution order."""

import polars as pl

from src.data.preprocess import remove_empty_or_duplicate

DF = pl.DataFrame({"text": ["a", "a", "", "b"], "label": [1, 1, 1, 0]})
rows = []  # shared list that tests append to


def test_removes_duplicates_and_empty():
    rows.append(remove_empty_or_duplicate(DF))
    assert rows[-1].height == 2


def test_only_one_result_recorded():
    # Passes alone, fails when run after the previous test: it sees that test's leftovers.
    assert len(rows) == 0
