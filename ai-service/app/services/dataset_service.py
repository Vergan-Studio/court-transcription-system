# ai-service/app/services/dataset_service.py

import os
import pandas as pd
from datasets import load_dataset, Audio
from pathlib import Path
from app.config import DATASETS_DIR
from app.config import AFRISPEECH_DIR


AFRISPEECH_PATH = DATASETS_DIR / "afrispeech"
FLEURS_PATH = DATASETS_DIR / "fleurs"
COMBINED_PATH = DATASETS_DIR / "combined"

AUDIO_DIR = COMBINED_PATH / "audio"
META_DIR = COMBINED_PATH / "metadata"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# AFRISPEECH
# ==================================================


def download_afrispeech():
    print("Checking AfriSpeech...")

    meta_path = AFRISPEECH_DIR / "metadata" / "train_processed.csv"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"""
AfriSpeech not ready.

Expected file:
{meta_path}

Fix:
→ Run dataset download step first OR
→ ensure AfriSpeech raw data is present
"""
        )

    df = pd.read_csv(meta_path)

    df = df.rename(columns={
        "text": "transcript",
        "audio_path": "audio"
    })

    df["dataset"] = "afrispeech"

    df.to_csv(META_DIR / "afrispeech.csv", index=False)

    print(f"AfriSpeech loaded: {len(df)} samples")

# ==================================================
# FLEURS (ENGLISH ONLY)
# ==================================================

def download_fleurs():
    print("Downloading FLEURS English...")

    ds = load_dataset(
        "google/fleurs",
        "en_us",
        split="train"
    )

    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    df = pd.DataFrame({
        "audio": [x["audio"]["path"] for x in ds],
        "transcript": [x["transcription"] for x in ds],
        "dataset": "fleurs"
    })

    df.to_csv(META_DIR / "fleurs.csv", index=False)

    print(f"FLEURS ready: {len(df)} samples")


# ==================================================
# COMBINE
# ==================================================

def build_combined_dataset():
    print("Building combined dataset...")

    af = pd.read_csv(META_DIR / "afrispeech.csv")
    fl = pd.read_csv(META_DIR / "fleurs.csv")

    df = pd.concat([af, fl], ignore_index=True)

    # shuffle for training stability
    df = df.sample(frac=1).reset_index(drop=True)

    # simple split
    train = df.sample(frac=0.8, random_state=42)
    temp = df.drop(train.index)
    val = temp.sample(frac=0.5, random_state=42)
    test = temp.drop(val.index)

    train.to_csv(META_DIR / "train.csv", index=False)
    val.to_csv(META_DIR / "val.csv", index=False)
    test.to_csv(META_DIR / "test.csv", index=False)

    print("Combined dataset ready")
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")


# ==================================================
# PIPELINE ENTRY
# ==================================================

def run_dataset_pipeline():
    download_afrispeech()
    download_fleurs()
    build_combined_dataset()