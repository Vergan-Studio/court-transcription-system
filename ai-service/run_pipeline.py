import sys
from google.colab import userdata

sys.path.append('/content/court-transcription-system/ai-service')

from dataset_builder import build_stratified_hf_datasets
from audio_preprocessor import get_whisper_processing_tools, prepare_dataset_mapping_function

def main():
    print("🚀 --- LAUNCHING MODULAR PREPROCESSING PIPELINE --- 🚀\n")
    
    try:
        HF_TOKEN = userdata.get('HF_TOKEN')
    except Exception:
        print("⚠️ Warning: 'HF_TOKEN' not found in Colab secrets. Attempting public stream...")
        HF_TOKEN = None
        
    # Phase 1: Build safe datasets via buffer streaming
    hf_datasets = build_stratified_hf_datasets(hf_token=HF_TOKEN)
    
    # Phase 2: Pull processors
    processor = get_whisper_processing_tools("openai/whisper-medium")
    mapping_fn = prepare_dataset_mapping_function(processor)
    
    # Phase 3: Execute map transformation
    print("\n⚡ Transforming raw waveforms into Whisper Medium Mel-Spectrogram features...")
    processed_datasets = hf_datasets.map(
        mapping_fn, 
        batched=True,
        batch_size=16,
        remove_columns=["audio", "sentence", "accent_group", "dataset_source"]
    )
    
    print("\n🏁 --- PIPELINE EXECUTION SUCCESSFUL --- 🏁")
    print(processed_datasets)
    print("\n📊 Verification Vector Confirmation:")
    print(f"   • Train feature dimension shape: {len(processed_datasets['train'][0]['input_features'])} log-Mel frequency bands.")

if __name__ == "__main__":
    main()
