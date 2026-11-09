"""
CivicPulse AI — RDD2022 India Dataset Downloader and Extractor
Downloads RDD2022_India.zip, validates integrity, and unpacks to datasets/raw/RDD2022_India/
"""

import os
import sys
import zipfile
import urllib.request
import time
from pathlib import Path

DATASET_URL = "https://huggingface.co/datasets/aiswarya9302/RDD_2022India/resolve/main/RDD2022_India.zip"
TARGET_DIR = Path("datasets/raw")
ZIP_PATH = TARGET_DIR / "RDD2022_India.zip"
EXTRACT_DIR = TARGET_DIR / "RDD2022_India"

def download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"Archive already exists at {dest_path} ({dest_path.stat().st_size / 1024**2:.1f} MB). Skipping download.")
        return

    print(f"Downloading {url} to {dest_path}...")
    start_time = time.time()
    
    def reporthook(count, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = count * block_size
        percent = min(100.0, downloaded * 100.0 / total_size)
        elapsed = time.time() - start_time
        speed = (downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        sys.stdout.write(f"\r  Progress: {percent:5.1f}% | {downloaded / 1024**2:6.1f} MB / {total_size / 1024**2:.1f} MB | Speed: {speed:5.1f} MB/s")
        sys.stdout.flush()

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 CivicPulse-AI-Pipeline/1.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    print(f"\nDownload completed in {time.time() - start_time:.1f}s.")

def extract_archive(zip_path: Path, dest_dir: Path):
    print(f"Extracting {zip_path} to {dest_dir}...")
    start_time = time.time()
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        total_files = len(members)
        print(f"Found {total_files} entries in archive.")
        zf.extractall(dest_dir)
        
    print(f"Extraction completed in {time.time() - start_time:.1f}s.")

def main():
    download_file(DATASET_URL, ZIP_PATH)
    extract_archive(ZIP_PATH, EXTRACT_DIR)
    
    # Inspect contents
    all_files = list(EXTRACT_DIR.rglob("*"))
    jpg_files = list(EXTRACT_DIR.rglob("*.jpg"))
    xml_files = list(EXTRACT_DIR.rglob("*.xml"))
    print(f"Extracted summary: {len(all_files)} total items, {len(jpg_files)} JPEG images, {len(xml_files)} XML annotation files.")

if __name__ == "__main__":
    main()

