from datasets import load_dataset
import evaluate
import pytest

from src.config import PROCESSED_DATA_DIR, SEED

ACC_THRESHOLD = 0.95


@pytest.fixture(scope="function")
def test_ds():
    ds = (
        load_dataset(
            "parquet",
            data_files={
                "test": str(PROCESSED_DATA_DIR / "test.parquet"),
            },
        )["test"]
        .shuffle(SEED)
        .select(range(1000))
    )
    return ds


def test_model_accuracy(pipe, test_ds):
    """
    Test the model's accuracy on a small test set.
    """
    accuracy = evaluate.load("accuracy")
    eval = evaluate.evaluator("text-classification")
    result = eval.compute(
        model_or_pipeline=pipe,
        data=test_ds,
        metric=accuracy,
        label_column="labels",
        label_mapping={"negative": 0, "positive": 1},
    )
    acc_score = result["accuracy"]
    assert acc_score > ACC_THRESHOLD, (
        f"Model accuracy is {acc_score:.2f}, which is below the threshold of {ACC_THRESHOLD}."
    )
