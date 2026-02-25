"""Download the barexam_qa dataset from HuggingFace.

Usage:
  uv run python download_data.py          # Download passages + QA splits
  uv run python download_data.py --check  # Check if data already exists

Downloads from: https://huggingface.co/datasets/reglab/barexam_qa
Directly downloads TSV/CSV files to bypass broken HF loading scripts.
"""

import os
import sys
import zipfile
import pandas as pd
from huggingface_hub import hf_hub_download

DATA_DIR = "datasets/barexam_qa"
PASSAGES_CSV = os.path.join(DATA_DIR, "barexam_qa_train.csv")
QA_CSV = os.path.join(DATA_DIR, "qa", "qa.csv")

REPO_ID = "reglab/barexam_qa"

def check_data():
    """Check if required data files exist."""
    files = {
        "Passages CSV": PASSAGES_CSV,
        "QA CSV": QA_CSV,
    }
    all_ok = True
    for label, path in files.items():
        exists = os.path.isfile(path)
        status = "OK" if exists else "MISSING"
        print(f"  {status}: {label} ({path})")
        if not exists:
            all_ok = False
    return all_ok

def download():
    """Download barexam_qa from HuggingFace and save to datasets/barexam_qa/."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "qa"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "passages"), exist_ok=True)

    # --- Passages (the main retrieval corpus) ---
    print(f"Downloading passages from {REPO_ID}...")
    
    # Passages come as a zipped TSV
    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="data/passages/train.tsv.zip",
        repo_type="dataset"
    )
    
    print(f"  Extracting passages...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
        # The zip contains 'train.tsv'
        tsv_path = os.path.join(DATA_DIR, "train.tsv")
        
    print(f"  Processing passages (TSV -> CSV)...")
    # Read TSV and save as CSV
    df_passages = pd.read_csv(tsv_path, sep='\t')
    df_passages.to_csv(PASSAGES_CSV, index=False)
    
    # Cleanup temp tsv
    if os.path.exists(tsv_path):
        os.remove(tsv_path)
    
    print(f"  Saved {len(df_passages)} passages to {PASSAGES_CSV}")

    # --- QA pairs (questions + gold passage indices) ---
    print(f"\nDownloading QA pairs from {REPO_ID}...")
    
    qa_splits = ["train", "validation", "test"]
    all_dfs = []
    
    for split in qa_splits:
        split_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"data/qa/{split}.csv",
            repo_type="dataset"
        )
        df_split = pd.read_csv(split_path)
        
        target_path = os.path.join(DATA_DIR, "qa", f"{split}.csv")
        df_split.to_csv(target_path, index=False)
        print(f"  {split}: {len(df_split)} rows -> {target_path}")
        all_dfs.append(df_split)

    # Create combined qa.csv
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(QA_CSV, index=False)
    print(f"  Combined all splits -> {QA_CSV} ({len(combined)} rows)")

    print(f"\nDone! Files saved to {DATA_DIR}/")
    print("\nNext steps:")
    print("  uv run python load_corpus.py curated   # Fast: ~1.5K passages, ~3 min")
    print("  uv run python load_corpus.py 20000      # Full: 20K passages, ~30 min")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        print("Checking data files:")
        if check_data():
            print("\nAll data files present.")
        else:
            print("\nSome files missing. Run: uv run python download_data.py")
        return

    if os.path.isfile(PASSAGES_CSV):
        print(f"Data already exists at {PASSAGES_CSV}.")
        print("Re-downloading will overwrite existing files.")
        resp = input("Continue? [y/N] ")
        if resp.lower() != "y":
            print("Cancelled.")
            return

    download()

if __name__ == "__main__":
    main()
