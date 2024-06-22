import os
import requests
from tqdm import tqdm

BASE_MODEL_DIR = "base_model"
BASE_FILES = {
    "dvae.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth",
    "mel_stats.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth"
}

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB

    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            progress_bar.update(size)

def download_base_files():
    if not os.path.exists(BASE_MODEL_DIR):
        os.makedirs(BASE_MODEL_DIR)

    for filename, url in BASE_FILES.items():
        filepath = os.path.join(BASE_MODEL_DIR, filename)
        if os.path.exists(filepath):
            print(f"{filename} already exists. Skipping download.")
        else:
            print(f"Downloading {filename}...")
            download_file(url, filepath)
            print(f"{filename} downloaded successfully.")

if __name__ == "__main__":
    download_base_files()
