
import os
import torch
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from torch.utils.data import Dataset
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, TaskType
import soundfile as sf
import json
from transformers.trainer_utils import get_last_checkpoint
from app.config import (
    BASE_MODEL,
    OUTPUT_DIR,
    TRAIN_CSV,
    VAL_CSV,
    VOCAB_PATH,
    MAX_STEPS,
    SAVE_STEPS,
    EVAL_STEPS,
    BATCH_SIZE,
    LEARNING_RATE,
    SAMPLE_RATE,
    RESUME_FROM_CHECKPOINT,
    CHECKPOINT_DIR
)


# ── Load court vocabulary prompt ──────────────────────────────────
def get_initial_prompt(tokenizer):
    """Load court vocabulary and create initial prompt tokens."""
    try:
        with open(VOCAB_PATH) as f:
            vocab = json.load(f)
        terms  = []
        for category, words in vocab.items():
            terms.extend(words)
        prompt = "Court proceedings transcript. Legal terminology: " + ", ".join(terms[:30])
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
        print(f"Court vocabulary prompt: {len(terms)} terms loaded")
        return prompt_ids
    except Exception as e:
        print(f"Vocabulary not found, proceeding without prompt: {e}")
        return None


# ── Dataset class ─────────────────────────────────────────────────
class AfriSpeechDataset(Dataset):
    def __init__(self, df, processor, prompt_ids=None):
        self.df         = df.reset_index(drop=True)
        self.processor  = processor
        self.prompt_ids = prompt_ids

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load audio
        try:
            audio, sr = sf.read(row["audio_path"])
        except Exception:
            # Return empty sample if file unreadable
            audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
            sr    = SAMPLE_RATE

        if audio is None or len(audio) == 0:
            audio = np.zeros(SAMPLE_RATE, dtype=np.float32)

        audio = audio.astype(np.float32)

        # Extract features
        input_features = self.processor.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        ).input_features[0]

        # Tokenize transcript
        labels = self.processor.tokenizer(
            str(row["transcript"]),
            return_tensors="pt"
        ).input_ids[0]

        return {
            "input_features": input_features,
            "labels":         labels
        }


# ── Data collator ─────────────────────────────────────────────────
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
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ── Load model with LoRA ──────────────────────────────────────────
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
        torch_dtype=torch.float32,
        device_map="auto"
    )
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens    = []
    model.config.use_cache                     = False

    # LoRA config
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


# ── Training function ─────────────────────────────────────────────
def run_finetuning():
    print("\n" + "="*50)
    print("  WHISPER FINE-TUNING V2")
    print("  (augmented data + validation + vocabulary)")
    print("="*50)

    # Load data
    print("\nLoading datasets...")
    df_train = pd.read_csv(TRAIN_CSV)
    df_val   = pd.read_csv(VAL_CSV)

    print(f"Training samples:   {len(df_train)}")
    print(f"Validation samples: {len(df_val)}")
    print(f"Augmented types:    {df_train['aug_type'].value_counts().to_dict() if 'aug_type' in df_train.columns else 'N/A'}")

    # Load model
    model, processor = load_model_with_lora()

    # Load vocabulary prompt
    prompt_ids = get_initial_prompt(processor.tokenizer)

    # Create datasets
    train_dataset = AfriSpeechDataset(df_train, processor, prompt_ids)
    val_dataset   = AfriSpeechDataset(df_val,   processor, prompt_ids)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Training arguments with validation
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
        dataloader_num_workers=2,
    )

    # Trainer with validation
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )

    # Train
    print("\nStarting training...")

    checkpoint = None

    if RESUME_FROM_CHECKPOINT:
        checkpoint = get_last_checkpoint(CHECKPOINT_DIR)

    if checkpoint is not None:
        print(f"🔁 Resuming from checkpoint: {checkpoint}")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        print("🆕 Starting fresh training")
        trainer.train()

    experiment = {
    "model": BASE_MODEL,
    "steps": MAX_STEPS,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    }

    with open(os.path.join(OUTPUT_DIR, "experiment.json"), "w") as f:
        json.dump(experiment, f, indent=2)

    # Save best model
    print("\nSaving best model...")
    model.save_pretrained(OUTPUT_DIR)
    processor. save_pretrained(OUTPUT_DIR)

    print("\n" + "="*50)
    print("  FINE-TUNING V2 COMPLETE")
    print(f"  Model saved: {OUTPUT_DIR}")
    print("="*50)

    return model, processor

