import os
import torch
import json
import numpy as np
import soundfile as sf
from functools import wraps
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from torch.utils.data import Dataset
from datasets import load_from_disk
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from transformers.trainer_utils import get_last_checkpoint
from app.config import (
    BASE_MODEL,
    OUTPUT_DIR,
    MAX_STEPS,
    SAVE_STEPS,
    EVAL_STEPS,
    BATCH_SIZE,
    LEARNING_RATE,
    RESUME_FROM_CHECKPOINT,
    CHECKPOINT_DIR
)

# ── Adaptive Dataset Wrapper ──────────────────────────────────────
class AdaptiveWhisperDataset(Dataset):
    def __init__(self, hf_dataset, processor):
        self.dataset = hf_dataset
        self.processor = processor
        
        cols = hf_dataset.column_names
        self.text_col = "transcript" if "transcript" in cols else ("sentence" if "sentence" in cols else "text")
        self.path_col = "audio_path" if "audio_path" in cols else ("path" if "path" in cols else None)
        self.has_audio_col = "audio" in cols

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        
        if "input_features" in row and "labels" in row:
            return {
                "input_features": row["input_features"],
                "labels": row["labels"]
            }
            
        audio = None
        sr = 16000
        
        if self.has_audio_col and isinstance(row["audio"], dict) and "array" in row["audio"]:
            audio = row["audio"]["array"]
            sr = row["audio"].get("sampling_rate", 16000)
        elif self.path_col and row[self.path_col]:
            try:
                audio, sr = sf.read(row[self.path_col])
            except Exception:
                pass
        
        if audio is None or len(audio) == 0:
            audio = np.zeros(16000, dtype=np.float32)
            sr = 16000
            
        audio = audio.astype(np.float32)
        
        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt"
        ).input_features[0]
        
        text_str = str(row[self.text_col]) if self.text_col in row else ""
        labels = self.processor.tokenizer(
            text_str,
            return_tensors="pt"
        ).input_ids[0]
        
        return {
            "input_features": input_features,
            "labels": labels
        }

# ── Data Collator ─────────────────────────────────────────────────
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        
        # ── CRITICAL FIX: TRUNCATE LABELS ──
        # Whisper max length is 448. We clamp to 448 to prevent crashes.
        labels = labels_batch["input_ids"]
        if labels.shape[1] > 448:
            labels = labels[:, :448]
        
        labels = labels.masked_fill(
            labels_batch.attention_mask[:, :labels.shape[1]].ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def load_model_with_lora():
    print("Loading Whisper processor...")
    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL,
        language="english",
        task="transcribe"
    )

    print("Loading Whisper model...")
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens    = []
    model.config.use_cache                     = False

    # ── THE ULTIMATE PEFT/WHISPER PATCH ──
    old_forward = model.forward
    @wraps(old_forward) # <-- Preserves signature to avoid KeyErrors
    def safe_forward(*args, **kwargs):
        # Strip ALL text-specific kwargs injected by PEFT
        kwargs.pop("input_ids", None)
        kwargs.pop("inputs_embeds", None)
        return old_forward(*args, **kwargs)
    model.forward = safe_forward

    print("Applying LoRA layers config...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor

def run_finetuning():
    print("\n" + "="*50)
    print("  WHISPER FINE-TUNING V2 (BULLETPROOF ENGINE)")
    print("="*50)

    processed_dataset_path = "/content/whisper_medium_processed_african_dataset"
    if not os.path.exists(processed_dataset_path):
        raise FileNotFoundError(f"❌ Processed dataset not found at {processed_dataset_path}. Please run ETL first!")

    print(f"\n📥 Loading dataset matrix from {processed_dataset_path}...")
    dataset_dict = load_from_disk(processed_dataset_path)
    
    model, processor = load_model_with_lora()

    print("Wrapping datasets with adaptive feature extraction...")
    train_dataset = AdaptiveWhisperDataset(dataset_dict["train"], processor)
    val_dataset   = AdaptiveWhisperDataset(dataset_dict["validation"], processor)

    print(f"✅ Loaded Training samples:   {len(train_dataset)}")
    print(f"✅ Loaded Validation samples: {len(val_dataset)}")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        max_steps=MAX_STEPS,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        report_to=["tensorboard"],
        predict_with_generate=False,
        push_to_hub=False,
        dataloader_num_workers=0,
        remove_unused_columns=False, # <-- Locked down
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )

    print("\n🚀 Launching training loop engine...")
    checkpoint = None
    if RESUME_FROM_CHECKPOINT:
        checkpoint = get_last_checkpoint(CHECKPOINT_DIR)

    if checkpoint is not None:
        print(f"🔁 Resuming from checkpoint: {checkpoint}")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        print("🆕 Starting fresh training run")
        trainer.train()

    experiment = {
        "model": BASE_MODEL,
        "steps": MAX_STEPS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
    }
    with open(os.path.join(OUTPUT_DIR, "experiment.json"), "w") as f:
        json.dump(experiment, f, indent=2)

    print("\n💾 Saving optimized best model weights...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print("\n" + "="*50)
    print("  FINE-TUNING V2 COMPLETE")
    print(f"  Model saved: {OUTPUT_DIR}")
    print("="*50)

    return model, processor
