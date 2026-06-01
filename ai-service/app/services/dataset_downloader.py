import os
import csv
from pathlib import Path
from datasets import load_dataset

from app.config import DATASETS_DIR

# ==================================================
# GLOBAL PATHS
# ==================================================

AFRISPEECH_DIR = DATASETS_DIR / "afrispeech"
FLEURS_DIR = DATASETS_DIR / "fleurs"
COMBINED_DIR = DATASETS_DIR / "combined"

META_DIR = COMBINED_DIR / "metadata"
AUDIO_DIR = COMBINED_DIR / "audio"

META_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# SAFETY CONFIG
# ==================================================

TARGET_LANGUAGE = "en"


# ==================================================
# 1. COMMON VOICE (STREAMING SAFE MODE)
# ==================================================

def stream_common_voice():
    """
    Streaming mode prevents:
    - disk overflow
    - broken HF cache
    - incomplete dataset downloads
    """

    print("📥 Streaming Common Voice (EN)...")

    dataset = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "en",
        split="train",
        streaming=True
    )

    return dataset


# ==================================================
# 2. FLEURS LOADER (CACHED SAFE MODE)
# ==================================================

def load_fleurs():
    print("📥 Loading FLEURS dataset (EN)...")

    dataset = load_dataset(
        "google/fleurs",
        "en_us",
        split="train"
    )

    return dataset


# ==================================================
# 3. AFRISPEECH LOCAL LOADER
# ==================================================

def load_afrispeech_local():
    print("📥 Loading AfriSpeech locally...")

    meta_file = AFRISPEECH_DIR / "metadata" / "train_processed.csv"

    if not meta_file.exists():
        raise FileNotFoundError(
            f"AfriSpeech metadata missing: {meta_file}"
        )

    return meta_file


# ==================================================
# 4. SAVE STREAMING DATA SAFELY
# ==================================================

def save_streaming_sample(sample, writer):
    """
    Normalized format:
    audio_path (if exists), transcript
    """

    try:
        text = sample.get("sentence") or sample.get("text")

        if not text:
            return

        writer.writerow({
            "audio_path": sample.get("path", ""),
            "transcript": text.strip(),
            "dataset": "common_voice"
        })

    except Exception:
        pass


# ==================================================
# 5. RUN PIPELINE (MAIN ENTRY)
# ==================================================

def run_dataset_download():
    """
    This is now your ONLY entry point for dataset acquisition.
    """

    print("\n==============================")
    print(" DATASET DOWNLOAD PIPELINE")
    print("==============================\n")

    output_file = META_DIR / "combined_raw.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["audio_path", "transcript", "dataset"]
        )
        writer.writeheader()

        # --------------------------
        # COMMON VOICE (STREAMING)
        # --------------------------
        try:
            cv = stream_common_voice()

            print("📡 Streaming samples...")
            for i, sample in enumerate(cv):
                save_streaming_sample(sample, writer)

                if i > 5000:  # safety cap for now
                    break

        except Exception as e:
            print(f"❌ Common Voice failed: {e}")

        # --------------------------
        # FLEURS
        # --------------------------
        try:
            fleurs = load_fleurs()

            for sample in fleurs:
                writer.writerow({
                    "audio_path": "",
                    "transcript": sample["transcription"],
                    "dataset": "fleurs"
                })

        except Exception as e:
            print(f"❌ FLEURS failed: {e}")

    print("\n==============================")
    print(" DATASET DOWNLOAD COMPLETE")
    print("==============================")
    print(f"Saved to: {output_file}")