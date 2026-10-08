from huggingface_hub import hf_hub_download
import polars as pl

from src.config import RAW_DATA_DIR

REPO_ID = "stanfordnlp/imdb"
SPLITS = {
    "train": "plain_text/train-00000-of-00001.parquet",
    "test": "plain_text/test-00000-of-00001.parquet",
}


def download_dataset():
    """
    Download the dataset from Hugging Face and save it to the 'data/raw' directory.
    """
    frames = [
        pl.read_parquet(hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset"))
        for filename in SPLITS.values()
    ]

    df = pl.concat(frames, how="vertical")
    df.write_parquet(RAW_DATA_DIR / "imdb.parquet")


if __name__ == "__main__":
    download_dataset()
