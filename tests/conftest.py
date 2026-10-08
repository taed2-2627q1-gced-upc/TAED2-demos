from collections.abc import Callable

import pytest
from transformers.pipelines import pipeline

from src.config import MODELS_DIR

MODEL_NAME = "distilbert-imdb"


@pytest.fixture(scope="session")
def pipe():
    """Load the model once and share it with every test that asks for it."""
    return pipeline("text-classification", str(MODELS_DIR / MODEL_NAME))


@pytest.fixture(scope="session")
def positive_score(pipe) -> Callable[[str], float]:
    """Factory fixture: returns a function mapping a text to the model's P(positive)."""

    def _positive_score(text: str) -> float:
        scores = pipe(text, top_k=None)
        scores = scores[0] if isinstance(scores[0], list) else scores
        return next(s["score"] for s in scores if s["label"] == "positive")

    return _positive_score
