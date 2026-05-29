
import os
import re
import json
import torch
import pandas as pd
import soundfile as sf
import sys
sys.path.insert(0, "/content/drive/MyDrive/speech-to-text-system/ai-service")
from app.utils.postprocessing import clean_transcript, check_audio_quality
from datetime import datetime
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from app.config import (
  BASE_MODEL,
  MODEL_DIR,
  RESULTS_DIR
)


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def wer(reference, hypothesis):
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()
    if len(ref) == 0:
        return 0.0
    d = [[0]*(len(hyp)+1) for _ in range(len(ref)+1)]
    for i in range(len(ref)+1): d[i][0] = i
    for j in range(len(hyp)+1): d[0][j] = j
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return round(d[len(ref)][len(hyp)] / len(ref) * 100, 2)


def load_finetuned_model():
    """Load fine-tuned model for evaluation. Works on CPU and GPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading fine-tuned model on {device}...")
    
    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL, language="english", task="transcribe"
    )
    
    if device == "cuda":
        base_model = WhisperForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        base_model = WhisperForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32
        )
    
    base_model.generation_config.forced_decoder_ids = None
    base_model.generation_config.suppress_tokens    = []

    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model.eval()
    print(f"Fine-tuned model loaded on {device}")
    return model, processor


def transcribe(model, processor, audio_path):
    """Transcribe a single audio file."""
    audio, sr = sf.read(audio_path)
    if len(audio) == 0:
        return ""
    inputs = processor.feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to("cuda").half()
    with torch.no_grad():
        predicted_ids = model.generate(input_features=inputs)
    result = processor.tokenizer.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0].strip()
    return clean_transcript(result)


def run_evaluation(test_csv, label="evaluation"):
    """
    Run full evaluation on a test set.
    Saves detailed results to Drive.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 50)
    print(f"  EVALUATION: {label.upper()}")
    print("=" * 50)

    df_test = pd.read_csv(test_csv)
    print(f"Test samples: {len(df_test)}")

    model, processor = load_finetuned_model()

    results = []
    for _, row in df_test.iterrows():
        hypothesis = transcribe(model, processor, row["audio_path"])
        error      = wer(row["transcript"], hypothesis)
        results.append({
            "accent":     row["accent"],
            "reference":  row["transcript"],
            "hypothesis": hypothesis,
            "wer":        error
        })
        print(f"Accent: {row['accent']:15} WER: {error}%")

    # Calculate summary
    df_results = pd.DataFrame(results)
    avg_wer    = round(df_results["wer"].mean(), 2)
    per_accent = df_results.groupby("accent")["wer"].mean().round(2).to_dict()

    summary = {
        "label":           label,
        "timestamp":       datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_samples":   len(df_test),
        "average_wer":     avg_wer,
        "per_accent_wer":  per_accent,
        "baselines": {
            "raw_audio":     35.22,
            "after_etl":     25.50,
            "after_finetune": 18.83
        }
    }

    # Save results
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(RESULTS_DIR, f"{label}_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    df_results.to_csv(
        os.path.join(RESULTS_DIR, f"{label}_{timestamp}_detail.csv"),
        index=False
    )

    print("\n" + "=" * 50)
    print(f"  AVERAGE WER:     {avg_wer}%")
    print(f"  BASELINE:        35.22%")
    print(f"  IMPROVEMENT:     {round(35.22 - avg_wer, 2)}%")
    print(f"  Results saved:   {results_path}")
    print("=" * 50)

    return summary
