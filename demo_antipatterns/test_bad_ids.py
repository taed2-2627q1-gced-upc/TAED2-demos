"""Antipattern 4: parametrize with objects and no ids. Failures are reported as opaque `case0`, `case1`..."""

import pytest
from transformers.pipelines import pipeline

from src.config import MODELS_DIR


@pytest.mark.parametrize(
    "case",
    [
        {"text": "This movie is great!", "expected": "positive"},
        {"text": "The movie was not bad.", "expected": "positive"},
        {"text": "I hated the plot and the characters.", "expected": "negative"},
    ],
)
def test_labels(case):
    pipe = pipeline("text-classification", str(MODELS_DIR / "distilbert-imdb"))
    assert pipe(case["text"])[0]["label"] == case["expected"]
