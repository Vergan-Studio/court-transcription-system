
import os
import json
from app.config import (
  COURT_VOCABULARY,
  SAVE_PATH
)


def save_vocabulary():
    """Save court vocabulary to Drive."""
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "w") as f:
        json.dump(COURT_VOCABULARY, f, indent=2)
    print(f"✅ Vocabulary saved: {SAVE_PATH}")
    return SAVE_PATH


def get_all_terms():
    """Get flat list of all vocabulary terms."""
    terms = []
    for category, words in COURT_VOCABULARY.items():
        terms.extend(words)
    return terms


def create_initial_prompt():
    """
    Create an initial prompt for Whisper that primes it
    with court vocabulary before transcribing.
    This is a simple but effective way to improve domain
    specific transcription without retraining.
    """
    terms = get_all_terms()
    # Whisper uses initial prompts to bias transcription
    prompt = (
        "Court proceedings transcript. Legal terminology: "
        + ", ".join(terms[:30])  # Whisper prompt limit
    )
    return prompt


def apply_vocabulary_to_transcription(model_dict, audio_path):
    """
    Transcribe with court vocabulary prompt.
    Works with both base and fine-tuned models.
    """
    import torch
    import soundfile as sf

    prompt = create_initial_prompt()

    if model_dict["type"] == "finetuned":
        processor = model_dict["processor"]
        model     = model_dict["model"]
        device    = model_dict["device"]

        audio, sr = sf.read(audio_path)
        inputs    = processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features

        if device == "cuda":
            inputs = inputs.to("cuda").half()

        # Encode prompt as decoder input
        prompt_ids = processor.tokenizer(
            prompt, return_tensors="pt"
        ).input_ids

        if device == "cuda":
            prompt_ids = prompt_ids.to("cuda")

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features=inputs,
                decoder_input_ids=prompt_ids[:, :4]  # use first few tokens
            )

        return processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

    else:
        # Base whisper — use initial_prompt parameter
        model  = model_dict["model"]
        result = model.transcribe(
            audio_path,
            language="en",
            initial_prompt=prompt,
            fp16=torch.cuda.is_available()
        )
        return result["text"].strip()


def run_vocabulary_setup():
    """Save vocabulary and print summary."""
    print("\n" + "=" * 50)
    print("  COURT VOCABULARY SETUP")
    print("=" * 50)

    save_vocabulary()
    terms = get_all_terms()
    prompt = create_initial_prompt()

    print(f"Total terms:      {len(terms)}")
    for category, words in COURT_VOCABULARY.items():
        print(f"{category}: {len(words)} terms")

    print(f"\nInitial prompt preview:")
    print(f"{prompt[:100]}...")
    print("\n" + "=" * 50)
    print("  VOCABULARY SETUP COMPLETE")
    print("=" * 50)

    return COURT_VOCABULARY
