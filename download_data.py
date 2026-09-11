"""
This script downloads the dataset from a specified URL and saves it to a local file.
"""

from pathlib import Path
from urllib.request import urlretrieve

DATA_URL = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
)
DATA_PATH = Path(__file__).with_name("the-verdict.txt")

def download_dataset():
    """
    Downloads the dataset from the specified URL and saves it to the local path.
    """
    urlretrieve(DATA_URL, DATA_PATH)
    print(f"Downloaded dataset to {DATA_PATH}")

if __name__ == "__main__":
    download_dataset()