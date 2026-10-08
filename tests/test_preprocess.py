import polars as pl
import pytest

from src.data.preprocess import remove_empty_or_duplicate, sanitize_text


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """A fresh, tiny DataFrame for every test, so no test can affect another."""
    return pl.DataFrame(
        {
            "text": ["great movie", "great movie", "great movie", "", None, "awful movie", "fine movie"],
            "label": [1, 1, 0, 1, 1, 0, None],
        }
    )


@pytest.mark.parametrize(
    "raw, expected",
    [
        pytest.param("  padded  ", "padded", id="strips-edges"),
        pytest.param("two  spaces", "two spaces", id="collapses-spaces"),
        pytest.param("line\nbreak", "line break", id="newline-to-space"),
        pytest.param("carriage\rreturn", "carriage return", id="carriage-return-to-space"),
        pytest.param("ﬁlm", "film", id="nfkd-ligature"),
        pytest.param(
            "a  b\nc  d",
            "a b c d",
            id="multiple-gaps",
        ),
    ],
)
def test_sanitize_text(raw, expected):
    result = sanitize_text(pl.DataFrame({"text": [raw]}))
    assert result["text"].to_list() == [expected]


def test_remove_empty_or_duplicate_drops_nulls_and_empty(sample_df):
    result = remove_empty_or_duplicate(sample_df)
    assert result["text"].null_count() == 0
    assert "" not in result["text"].to_list()
    assert result["label"].null_count() == 0


def test_remove_empty_or_duplicate_drops_exact_duplicates(sample_df):
    result = remove_empty_or_duplicate(sample_df)
    assert result.filter((pl.col("text") == "great movie") & (pl.col("label") == 1)).height == 1


def test_remove_empty_or_duplicate_keeps_same_text_with_different_label(sample_df):
    result = remove_empty_or_duplicate(sample_df)
    assert sorted(result.filter(pl.col("text") == "great movie")["label"].to_list()) == [0, 1]
