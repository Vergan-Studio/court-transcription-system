import sys
import os
from transformers import WhisperProcessor

# Append directory to paths to allow modular importing
sys.path.append('/content/court-transcription-system/ai-service')
from dataset_builder import build_stratified_hf_datasets

def main():
    print("🚀 --- LAUNCHING PRODUCTION ETL FEATURE MAPPER --- 🚀")
    
    # 1. Run our newly created structural dataset builder code
    hf_datasets = build_stratified_hf_datasets(hf_token=None)
    
    # 2. Load the pipeline processor assets
    print("\n📥 Loading pretrained Whisper processing assets...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="en", task="transcribe")
    
    # 3. Vectorization wrapper logic
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    print("\n⚡ Transforming raw audio inputs into Whisper Medium Mel-Spectrogram blocks...")
    processed_datasets = hf_datasets.map(prepare_dataset, remove_columns=hf_datasets["train"].column_names)
    
    # 4. Save to target storage directory
    output_dir = "/content/whisper_medium_processed_african_dataset"
    processed_datasets.save_to_disk(output_dir)
    print(f"\n💾 SUCCESS! Advanced ETL clean features written to disk at: {output_dir}")

if __name__ == "__main__":
    main()
