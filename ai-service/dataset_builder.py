import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

def audit_audio_and_text(file_path, raw_text, target_sr=16000):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return None
    cleaned_text = str(raw_text).strip()
    if not cleaned_text or cleaned_text.lower() in ['nan', 'none', '']:
        return None

    try:
        array, sr = librosa.load(file_path, sr=target_sr)
        duration = len(array) / target_sr
        if duration < 0.5 or duration > 30.0:
            return None
        if np.max(np.abs(array)) < 1e-4 or np.std(array) == 0:
            return None
        return {"array": array, "sampling_rate": target_sr}, cleaned_text
    except Exception:
        return None

def build_stratified_hf_datasets(hf_token=None):
    print("\n🌍 --- STARTING PRODUCTION DATA COMPILATION PIPELINE --- 🌍")
    unified_list = []
    
    # LAYER 1: Phase 0 Repository Data
    print("📦 Auditing Layer 1: Phase 0 Repository Data...")
    phase0_csv = '/content/court-transcription-system/ai-service/phase0_dataset.csv'
    afrispeech_dir = '/content/afrispeech_data'
    l1_count = 0
    
    if os.path.exists(phase0_csv):
        df_phase0 = pd.read_csv(phase0_csv)
        for _, row in df_phase0.iterrows():
            audio_path = row['audio_path']
            if not os.path.exists(audio_path):
                audio_path = os.path.join(afrispeech_dir, os.path.basename(audio_path))
            result = audit_audio_and_text(audio_path, row['transcript'])
            if result:
                audio_feat, text = result
                unified_list.append({
                    'audio': audio_feat,
                    'sentence': text,
                    'accent_group': 'west_african',
                    'dataset_source': 'phase0_repo'
                })
                l1_count += 1
    print(f"   ✅ Added {l1_count} pristine Phase 0 records.")

    # LAYER 2: Ghanaian Accent Data (INTEGRATING PSEUDO-TRANSCRIPTS)
    print("📦 Auditing Layer 2: Ghanaian Accent Pseudo-Transcripts...")
    ghana_csv = '/content/ghana_pseudo_transcripts.csv'
    l2_count = 0
    
    if os.path.exists(ghana_csv):
        df_ghana = pd.read_csv(ghana_csv)
        for _, row in df_ghana.iterrows():
            audio_path = row['audio_path']
            result = audit_audio_and_text(audio_path, row['transcript'])
            if result:
                audio_feat, text = result
                unified_list.append({
                    'audio': audio_feat,
                    'sentence': text,
                    'accent_group': 'west_african',
                    'dataset_source': 'ghana_accent'
                })
                l2_count += 1
    print(f"   ✅ Added {l2_count} pristine Ghanaian Accent records.")

    # LAYER 3: Common Voice Dataset
    print("📦 Auditing Layer 3: Common Voice African Cache...")
    cv_cache_dir = "/content/common_voice_african_local"
    cv_csv = os.path.join(cv_cache_dir, "african_manifest.csv")
    l3_count = 0
    
    if os.path.exists(cv_csv):
        df_cv = pd.read_csv(cv_csv)
        for _, row in df_cv.iterrows():
            audio_path = os.path.join(cv_cache_dir, row['filename'])
            result = audit_audio_and_text(audio_path, row['text'])
            if result:
                audio_feat, text = result
                unified_list.append({
                    'audio': audio_feat,
                    'sentence': text,
                    'accent_group': 'southern_african',
                    'dataset_source': 'common_voice'
                })
                l3_count += 1
    print(f"   ✅ Added {l3_count} pristine Common Voice African records.")

    total_records = len(unified_list)
    if total_records == 0:
        raise ValueError("❌ Critical Error: No valid samples found.")
        
    print(f"\n⚡ Total pristine records entering stratification: {total_records}")
    
    meta_records = [{'index_id': idx, 'accent_group': item['accent_group']} for idx, item in enumerate(unified_list)]
    df_meta = pd.DataFrame(meta_records)
    
    train_idx, temp_idx = train_test_split(df_meta, test_size=0.20, random_state=42, stratify=df_meta['accent_group'])
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, stratify=temp_idx['accent_group'])
    
    datasets = DatasetDict({
        "train": Dataset.from_list([unified_list[i] for i in train_idx['index_id']]),
        "validation": Dataset.from_list([unified_list[i] for i in val_idx['index_id']]),
        "test": Dataset.from_list([unified_list[i] for i in test_idx['index_id']])
    })
    
    print("🏁 --- PRODUCTION BALANCED COMPILATION COMPLETELY SECURED --- 🏁")
    return datasets
