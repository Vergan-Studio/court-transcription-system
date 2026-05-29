
import sys
import os
import pandas as pd
sys.path.insert(0, "/content/drive/MyDrive/speech-to-text-system/ai-service")

METADATA_PATH  = "/content/drive/MyDrive/datasets/afrispeech/metadata/train_processed.csv"
TRAIN_SPLIT    = "/content/drive/MyDrive/datasets/afrispeech/metadata/train_split.csv"
VAL_SPLIT      = "/content/drive/MyDrive/datasets/afrispeech/metadata/val_split.csv"

# Only run full ETL if processed metadata doesnt exist
if os.path.exists(METADATA_PATH):
    print("Processed metadata already exists - skipping ETL transform")
    print(f"Loading: {METADATA_PATH}")
    df_processed = pd.read_csv(METADATA_PATH)
    print(f"Loaded {len(df_processed)} samples")
else:
    print("Running full ETL pipeline...")
    from app.services.etl_service import run_etl
    df_processed = run_etl(split="train")

# Always create train/val split if it doesnt exist
if os.path.exists(TRAIN_SPLIT) and os.path.exists(VAL_SPLIT):
    print("Train/val split already exists - skipping")
    df_train = pd.read_csv(TRAIN_SPLIT)
    df_val   = pd.read_csv(VAL_SPLIT)
    print(f"Train: {len(df_train)} samples")
    print(f"Val:   {len(df_val)} samples")
else:
    print("Creating train/val split...")
    from app.services.etl_service import create_train_val_split
    df_train, df_val = create_train_val_split(df_processed)
