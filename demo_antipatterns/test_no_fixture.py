"""Antipattern 1: no fixture. The model is reloaded in every test, which makes the suite slow."""

from transformers.pipelines import pipeline

from src.config import MODELS_DIR


def test_positive():
    pipe = pipeline("text-classification", str(MODELS_DIR / "distilbert-imdb"))
    assert pipe("This movie is great!")[0]["label"] == "positive"


def test_negative():
    pipe = pipeline("text-classification", str(MODELS_DIR / "distilbert-imdb"))
    assert pipe("I hated the plot and the characters.")[0]["label"] == "negative"


def test_strong_negative():
    pipe = pipeline("text-classification", str(MODELS_DIR / "distilbert-imdb"))
    assert pipe("Terrible acting, a waste of time.")[0]["label"] == "negative"
