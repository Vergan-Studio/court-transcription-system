
import re
import os
import json
import numpy as np
import soundfile as sf
from app.config import (
  PUNCTUATION_RULES,
  MEDICAL_FIXES
)


def clean_transcript(text: str) -> str:
    """
    Apply post-processing rules to clean transcript text.
    - Replace spoken punctuation with symbols
    - Fix common medical term errors
    - Clean up whitespace
    """
    if not text:
        return text

    # Apply punctuation rules
    for pattern, replacement in PUNCTUATION_RULES.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Apply medical fixes
    for pattern, replacement in MEDICAL_FIXES.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Clean up whitespace
    text = re.sub(r" +", " ", text)
    text = re.sub(r" \.", ".", text)
    text = re.sub(r" ,", ",", text)
    text = text.strip()

    return text


# ── Audio quality check ───────────────────────────────────────────
def check_audio_quality(audio_path: str, min_db=-40.0) -> bool:
    """
    Check if audio has enough signal to be worth transcribing.
    Returns True if audio quality is acceptable.
    """
    try:
        audio, sr = sf.read(audio_path)
        if len(audio) == 0:
            return False

        # Calculate RMS energy in dB
        rms    = np.sqrt(np.mean(audio**2))
        if rms == 0:
            return False
        db     = 20 * np.log10(rms)

        return db > min_db
    except Exception:
        return False


def run_postprocessing():
    """Test post-processing on sample transcripts."""
    print("\n" + "="*50)
    print("  POST-PROCESSING RULES TEST")
    print("="*50)

    test_cases = [
        "We decided full stop because if we are going comma we may as well get it back full stop",
        "Respiratory rate is 48 circles per minute with bilateral crass crepitations next line",
        "The patient was acian nose and afebri on examination comma next line",
    ]

    for text in test_cases:
        cleaned = clean_transcript(text)
        print(f"Before: {text}")
        print(f"After:  {cleaned}")
        print()

    print("="*50)
    print("  POST-PROCESSING READY")
    print("="*50)
