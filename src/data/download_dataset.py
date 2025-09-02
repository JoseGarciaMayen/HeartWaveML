import os
import requests
import zipfile
from io import BytesIO

MITBIH_URL = "https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip"

DEST_FOLDER = os.path.join("data", "raw")

def download_and_extract():
    os.makedirs(DEST_FOLDER, exist_ok=True)

    if os.path.exists(os.path.join(DEST_FOLDER, "mit-bih-arrhythmia-database-1.0.0")):
        print("Dataset already exists. Skipping download.")
        return
    else:
        print(f"Downloading dataset from: {MITBIH_URL} ...")
        response = requests.get(MITBIH_URL, stream=True)
        response.raise_for_status()

        print("Downloaded dataset. Extracting archives...")
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(DEST_FOLDER)

        print("Extraction ended. Dataset downloaded and extracted at data/raw.")

if __name__ == "__main__":
    download_and_extract()