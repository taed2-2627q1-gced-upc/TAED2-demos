"""Antipattern 3: copy-pasted checks in one test. The first failure hides every case after it."""

from transformers.pipelines import pipeline

from src.config import MODELS_DIR


def test_simple_sentences():
    pipe = pipeline("text-classification", str(MODELS_DIR / "distilbert-imdb"))
    assert pipe("This movie is great!")[0]["label"] == "positive"
    assert pipe("The movie was not bad.")[0]["label"] == "positive"  # fails; the next lines never run
    assert pipe("I hated the plot and the characters.")[0]["label"] == "negative"
    assert pipe("Terrible acting, a waste of time.")[0]["label"] == "negative"
