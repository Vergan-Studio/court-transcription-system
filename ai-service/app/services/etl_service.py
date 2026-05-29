
import os
import json
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
from app.config import (
  RAW_DIR,
  PROCESSED_DIR,
  METADATA_DIR,
  TRAIN_CSV,
  TEST_CSV,
  MIN_DURATION,
  MAX_DURATION,
  SAMPLE_RATE,
  TARGET_DB)


# ── Step 1: Extract ───────────────────────────────────────────────
def extract(split="train"):
    """
    Find all downloaded audio files and match with transcripts.
    Returns a dataframe of matched audio/transcript pairs.
    """
    print(f"\n[EXTRACT] Finding downloaded audio files...")
    
    # Load transcript CSV
    csv_path = TRAIN_CSV if split == "train" else TEST_CSV
    df = pd.read_csv(csv_path)
    df["wav_filename"] = df["audio_paths"].apply(
        lambda x: x.split("/")[-1].replace(".wav", "")
    )
    
    # Find all downloaded wav files
    downloaded = {}
    for root, dirs, files in os.walk(RAW_DIR):
        if "processed" in root or "metadata" in root:
            continue  # skip processed folder
        for f in files:
            if f.endswith(".wav"):
                audio_id = f.replace(".wav", "")
                downloaded[audio_id] = os.path.join(root, f)
    
    print(f"[EXTRACT] Found {len(downloaded)} audio files")
    
    # Match with transcripts
    matches = df[df["wav_filename"].isin(downloaded.keys())].copy()
    matches["raw_audio_path"] = matches["wav_filename"].map(downloaded)
    
    print(f"[EXTRACT] Matched {len(matches)} audio/transcript pairs")
    print(f"[EXTRACT] Accents: {matches['accent'].value_counts().to_dict()}")
    
    return matches


# ── Step 2: Transform ─────────────────────────────────────────────
def get_duration(file_path):
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        file_path
    ], capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0.0


def get_volume(file_path):
    """Get mean volume in dBFS using ffmpeg."""
    result = subprocess.run([
        "ffmpeg", "-i", file_path,
        "-af", "volumedetect",
        "-f", "null", "-"
    ], capture_output=True, text=True)
    for line in result.stderr.split("\n"):
        if "mean_volume" in line:
            try:
                return float(line.split(":")[-1].replace("dB", "").strip())
            except:
                return None
    return None


def normalize_audio(input_path, output_path, target_db=TARGET_DB):
    """
    Convert audio to:
    - mono
    - 16kHz
    - wav format
    - normalized volume
    - trimmed silence
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Calculate volume adjustment
    current_db = get_volume(input_path)
    if current_db is not None:
        volume_adjust = target_db - current_db
        volume_filter = f"volume={volume_adjust}dB,silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB"
    else:
        volume_filter = "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB"
    
    command = [
        "ffmpeg",
        "-i", input_path,
        "-ac", "1",              # mono
        "-ar", str(SAMPLE_RATE), # 16kHz
        "-c:a", "flac",
        "-af", volume_filter,    # normalize + trim silence
        "-y",                    # overwrite
        output_path
    ]
    
    result = subprocess.run(command, capture_output=True)
    return result.returncode == 0


def clean_transcript(text):
    """
    Clean transcript text:
    - Remove extra whitespace
    - Normalize apostrophes
    - Remove special characters
    - Keep letters, numbers, spaces, basic punctuation
    """
    import re
    text = text.strip()
    text = text.replace("w/", "with")
    text = text.replace("\n", " ")
    text = re.sub(r"[^a-zA-Z0-9\s\.,!?\'-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def transform(df):
    """
    Process all audio files:
    - Filter by duration
    - Normalize audio
    - Clean transcripts
    - Skip corrupted files
    """
    print(f"\n[TRANSFORM] Processing {len(df)} files...")
    
    processed = []
    skipped   = {"too_short": 0, "too_long": 0, "corrupted": 0, "empty_transcript": 0}
    
    for i, row in df.iterrows():
        raw_path = row["raw_audio_path"]
        accent   = row["accent"]
        
        # Check duration
        duration = get_duration(raw_path)
        if duration < MIN_DURATION:
            skipped["too_short"] += 1
            continue
        if duration > MAX_DURATION:
            skipped["too_long"] += 1
            continue
        
        # Clean transcript
        transcript = clean_transcript(str(row["transcript"]))
        if len(transcript) < 5:
            skipped["empty_transcript"] += 1
            continue
        
        # Set output path
        filename    = os.path.basename(raw_path)
        filename = filename.replace(".wav", ".flac")
        output_path = os.path.join(PROCESSED_DIR, accent, filename)
        
        # Normalize audio
        success = normalize_audio(raw_path, output_path)
        if not success:
            skipped["corrupted"] += 1
            continue
        
        # Verify output exists and has content
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            skipped["corrupted"] += 1
            continue
        
        processed.append({
            "audio_path":  output_path,
            "transcript":  transcript,
            "accent":      accent,
            "duration":    duration,
            "raw_path":    raw_path
        })
        
        if len(processed) % 50 == 0:
            print(f"[TRANSFORM] Processed {len(processed)}/{len(df)} files...")
    
    print(f"\n[TRANSFORM] ✅ Processed: {len(processed)}")
    print(f"[TRANSFORM] ⏭️  Skipped:   {skipped}")
    
    return pd.DataFrame(processed)


# ── Step 3: Load ──────────────────────────────────────────────────
def load(df_processed, split="train"):
    """
    Save processed metadata and statistics to Drive.
    """
    print(f"\n[LOAD] Saving metadata...")
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Remove duplicates
    before = len(df_processed)
    df_processed = df_processed.drop_duplicates(
        subset=["audio_path"], keep="first"
    ).reset_index(drop=True)
    print(f"[LOAD] Removed {before - len(df_processed)} duplicates")

    # Add proper naming
    def make_proper_name(row, idx):
        accent  = row["accent"].replace(" ", "_").replace("/", "_")
        clip_id = str(idx).zfill(4)
        return f"{accent}_{clip_id}.wav"

    df_processed["proper_name"] = [
        make_proper_name(row, i) for i, row in df_processed.iterrows()
    ]

    # Save processed metadata
    meta_path = os.path.join(METADATA_DIR, f"{split}_processed.csv")
    df_processed.to_csv(meta_path, index=False)
    print(f"[LOAD] ✅ Metadata saved: {meta_path}")
    
    # Save statistics
    stats = {
        "created_at":      datetime.now().strftime("%Y%m%d_%H%M%S"),
        "split":           split,
        "total_samples":   len(df_processed),
        "total_hours":     round(df_processed["duration"].sum() / 3600, 2),
        "avg_duration":    round(df_processed["duration"].mean(), 2),
        "accents":         df_processed["accent"].value_counts().to_dict(),
        "min_duration":    round(df_processed["duration"].min(), 2),
        "max_duration":    round(df_processed["duration"].max(), 2),
    }
    
    stats_path = os.path.join(METADATA_DIR, f"{split}_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"[LOAD] ✅ Stats saved: {stats_path}")
    print(f"\n── DATASET STATISTICS ──────────────────────")
    print(f"Total samples:  {stats['total_samples']}")
    print(f"Total hours:    {stats['total_hours']}")
    print(f"Avg duration:   {stats['avg_duration']}s")
    print(f"Accents:        {stats['accents']}")
    print(f"────────────────────────────────────────────")
    
    return meta_path




# ── Step 4: Train/Val Split ───────────────────────────────────────
def create_train_val_split(df, val_size=0.2, random_state=42):
    """
    Split processed dataset into train and validation sets.
    Stratified by accent to ensure each accent is represented in both.
    """
    from sklearn.model_selection import train_test_split

    print(f"\n[SPLIT] Creating train/val split...")
    print(f"[SPLIT] Total samples: {len(df)}")

    # Handle accents with only 1 sample — cant stratify those
    accent_counts = df["accent"].value_counts()
    single_sample = accent_counts[accent_counts == 1].index.tolist()
    
    df_single   = df[df["accent"].isin(single_sample)]
    df_stratify = df[~df["accent"].isin(single_sample)]

    if len(df_stratify) > 0:
        df_train, df_val = train_test_split(
            df_stratify,
            test_size=val_size,
            random_state=random_state,
            stratify=df_stratify["accent"]
        )
        # Add single sample accents to training
        df_train = pd.concat([df_train, df_single]).reset_index(drop=True)
    else:
        df_train, df_val = train_test_split(
            df,
            test_size=val_size,
            random_state=random_state
        )

    df_val = df_val.reset_index(drop=True)

    print(f"[SPLIT] Training samples:   {len(df_train)}")
    print(f"[SPLIT] Validation samples: {len(df_val)}")

    # Save splits
    train_path = os.path.join(METADATA_DIR, "train_split.csv")
    val_path   = os.path.join(METADATA_DIR, "val_split.csv")

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)

    print(f"[SPLIT] ✅ Train split saved: {train_path}")
    print(f"[SPLIT] ✅ Val split saved:   {val_path}")

    return df_train, df_val


# ── Full ETL Pipeline (updated) ───────────────────────────────────
def run_etl(split="train"):
    print("=" * 50)
    print("  ETL PIPELINE")
    print("=" * 50)

    # Extract
    df_raw = extract(split)

    # Transform
    df_processed = transform(df_raw)

    # Load
    meta_path = load(df_processed, split)

    # Create train/val split only for training data
    if split == "train":
        df_train, df_val = create_train_val_split(df_processed)

    print("\n" + "=" * 50)
    print("  ETL COMPLETE")
    print(f"  Ready for fine-tuning: {meta_path}")
    print("=" * 50)

    return df_processed

