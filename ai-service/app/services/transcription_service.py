
import os
import torch
import whisper
import subprocess
import tempfile
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from app.config import (
  BASE_MODEL,
  MODEL_DIR,
  SAMPLE_RATE
)


def load_whisper_model(model_size: str = "medium.en"):
    """
    Load Whisper model.
    Uses fine-tuned model if available, otherwise falls back to base model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(MODEL_DIR):
        print(f"Loading fine-tuned Whisper model on {device}...")
        
        processor = WhisperProcessor.from_pretrained(
            BASE_MODEL, language="english", task="transcribe"
        )
        base_model = WhisperForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        base_model.generation_config.forced_decoder_ids = None
        base_model.generation_config.suppress_tokens    = []

        model = PeftModel.from_pretrained(base_model, MODEL_DIR)
        model.eval()

        print(f"Fine-tuned Whisper loaded on {device}")
        return {"type": "finetuned", "model": model, "processor": processor, "device": device}

    else:
        print(f"Fine-tuned model not found. Loading base Whisper {model_size} on {device}...")
        model = whisper.load_model(model_size, device=device)
        print(f"Base Whisper {model_size} loaded on {device}")
        return {"type": "base", "model": model, "device": device}


def transcribe_segment(model_dict, audio_path: str, start: float, end: float) -> str:
    """
    Transcribe a specific segment of audio between start and end times.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    command = [
        "ffmpeg", "-i", audio_path,
        "-ss", str(start),
        "-to", str(end),
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-acodec", "pcm_s16le",
        "-y", tmp_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    result = _transcribe_file(model_dict, tmp_path)
    os.remove(tmp_path)
    return result


def transcribe_full(model_dict, audio_path: str) -> str:
    """
    Transcribe a full audio file.
    """
    print(f"Transcribing: {audio_path}")
    result = _transcribe_file(model_dict, audio_path)
    print(f"Transcription complete")
    return result


def _transcribe_file(model_dict, audio_path: str) -> str:
    """
    Internal transcription function that handles both model types.
    """
    if model_dict["type"] == "finetuned":
        audio, sr = sf.read(audio_path)
        if len(audio) == 0:
            return ""

        processor = model_dict["processor"]
        model     = model_dict["model"]
        device    = model_dict["device"]

        inputs = processor.feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        ).input_features

        if device == "cuda":
            inputs = inputs.to("cuda").half()
        else:
            inputs = inputs.to("cpu")

        with torch.no_grad():
            predicted_ids = model.generate(input_features=inputs)

        return processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

    else:
        # Base whisper model
        model = model_dict["model"]
        result = model.transcribe(
            audio_path,
            language="en",
            fp16=torch.cuda.is_available()
        )
        return result["text"].strip()
