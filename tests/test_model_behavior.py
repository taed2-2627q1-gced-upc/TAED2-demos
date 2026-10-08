"""
Behavioral tests in the style of CheckList (Ribeiro et al., 2020):
minimal functionality (MFT), invariance (INV) and directional expectation (DIR).
"""

import pytest

DIR_NEGATIVE_MARGIN = 0.1
DIR_POSITIVE_TOLERANCE = 0.01


@pytest.mark.parametrize(
    "text, expected_label",
    [
        pytest.param("This movie is great!", "positive", id="simple-positive"),
        pytest.param("It was a fantastic film.", "positive", id="synonym-positive"),
        pytest.param("I hated the plot and the characters.", "negative", id="simple-negative"),
        pytest.param("Terrible acting, a waste of time.", "negative", id="strong-negative"),
        pytest.param(
            "The movie was not bad.",
            "positive",
            id="negation",
            marks=pytest.mark.xfail(strict=True, reason="Model ignores negation of a negative word"),
        ),
    ],
)
def test_mft_simple_sentences(pipe, text, expected_label):
    """MFT: simple, unambiguous inputs must be classified correctly."""
    result = pipe(text)
    assert result[0]["label"] == expected_label, (
        f"Model predicted {result[0]['label']} for '{text}', expected {expected_label}."
    )


@pytest.mark.parametrize(
    "original, perturbed",
    [
        pytest.param("John loved this movie.", "Maria loved this movie.", id="name-swap"),
        pytest.param("I loved this movie.", "I LOVED THIS MOVIE.", id="casing"),
        pytest.param("I loved this movie.", "I loved this movie. I watched it on Tuesday.", id="neutral-suffix"),
        pytest.param(
            "I loved this movie.",
            "I lovd this movie.",
            id="typo",
            marks=pytest.mark.xfail(strict=True, reason="A single typo flips the prediction"),
        ),
    ],
)
def test_inv_label_does_not_change(pipe, original, perturbed):
    """INV: changes that should not matter must not change the predicted label."""
    assert pipe(original)[0]["label"] == pipe(perturbed)[0]["label"], (
        f"Label changed after perturbing '{original}' into '{perturbed}'."
    )


@pytest.mark.parametrize(
    "base, suffix, direction",
    [
        pytest.param("I loved the acting.", " However, the ending was terrible.", "down", id="negative-suffix"),
        pytest.param("I loved the acting.", " The ending was fantastic.", "up", id="positive-suffix"),
    ],
)
def test_dir_score_moves_in_expected_direction(positive_score, base, suffix, direction):
    """DIR: adding sentiment-bearing text must push P(positive) in the expected direction."""
    before = positive_score(base)
    after = positive_score(base + suffix)
    if direction == "down":
        assert after < before - DIR_NEGATIVE_MARGIN, f"P(positive) went from {before:.2f} to {after:.2f}."
    else:
        assert after >= before - DIR_POSITIVE_TOLERANCE, f"P(positive) went from {before:.2f} to {after:.2f}."
