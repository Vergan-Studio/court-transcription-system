from transformers import WhisperProcessor

def get_whisper_processing_tools(model_id="openai/whisper-medium"):
    print(f"📥 Loading pretrained Whisper processing assets for: {model_id}")
    processor = WhisperProcessor.from_pretrained(model_id, language="english", task="transcribe")
    return processor

def prepare_dataset_mapping_function(processor):
    def batch_preprocessing_fn(batch):
        # 1. Extract audio arrays and automatically resample via HF Audio features
        audio_inputs = [sample["array"] for sample in batch["audio"]]
        sampling_rate = batch["audio"][0]["sampling_rate"]
        
        # Compute 80-channel log-Mel spectrograms padded/truncated to 30 seconds
        input_features = processor.feature_extractor(
            audio_inputs, 
            sampling_rate=sampling_rate, 
            return_tensors="pt"
        ).input_features
        
        # 2. Tokenize English text sentences into target label IDs
        labels = processor.tokenizer(batch["sentence"]).input_ids
        
        # Return processed dictionary tensors
        batch["input_features"] = input_features
        batch["labels"] = labels
        return batch
        
    return batch_preprocessing_fn
