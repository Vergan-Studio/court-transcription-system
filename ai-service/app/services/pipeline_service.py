
import json
import os
from datetime import datetime

from app.utils.audio_utils import preprocess_audio, validate_audio
from app.utils.postprocessing import clean_transcript, check_audio_quality
from app.services.diarization_service import load_diarization_pipeline, run_diarization
from app.services.transcription_service import load_whisper_model, transcribe_segment
from app.config import MIN_SEGMENT_DURATION


def run_pipeline(audio_path: str, hf_token: str, output_dir: str) -> dict:
    """
    Full court transcription pipeline.
    """

    # ── Step 1: Validate and preprocess ──────────────────────────────
    print("\n[1/5] Validating audio...")
    if not validate_audio(audio_path):
        raise FileNotFoundError(f"Audio file not found or empty: {audio_path}")

    preprocessed_path = os.path.join(
        output_dir,
        os.path.basename(audio_path).replace(".wav", "_preprocessed.wav")
    )
    preprocess_audio(audio_path, preprocessed_path)

    # ── Step 2: Load models ───────────────────────────────────────────
    print("\n[2/5] Loading models...")
    diarization_pipeline = load_diarization_pipeline(hf_token)
    whisper_model        = load_whisper_model("medium.en")

    # ── Step 3: Run diarization ───────────────────────────────────────
    print("\n[3/5] Running diarization...")
    segments = run_diarization(diarization_pipeline, preprocessed_path)

    # Filter out segments that are too short
    before = len(segments)
    segments = [
        s for s in segments
        if (s["end"] - s["start"]) >= MIN_SEGMENT_DURATION
    ]
    print(f"✅ Filtered {before - len(segments)} short segments "
          f"— {len(segments)} remaining")

    # ── Step 4: Transcribe each segment ──────────────────────────────
    print("\n[4/5] Transcribing segments...")
    transcript = []

    for i, segment in enumerate(segments):
        print(f"  Segment {i+1}/{len(segments)} "
              f"[{segment['start']}s → {segment['end']}s] "
              f"{segment['speaker']}")

        # Check audio quality before transcribing
        if not check_audio_quality(preprocessed_path):
            print(f"  Skipping segment - poor audio quality")
            continue

        text = transcribe_segment(
            whisper_model,
            preprocessed_path,
            segment["start"],
            segment["end"]
        )

        if text:
            transcript.append({
                "speaker": segment["speaker"],
                "start":   segment["start"],
                "end":     segment["end"],
                "text":    text
            })

    # ── Step 5: Save output ───────────────────────────────────────────
    print("\n[5/5] Saving transcript...")
    os.makedirs(output_dir, exist_ok=True)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"transcript_{timestamp}.json")

    result = {
        "audio_file":  os.path.basename(audio_path),
        "created_at":  timestamp,
        "total_turns": len(transcript),
        "transcript":  transcript
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ Transcript saved: {output_path}")
    print(f"✅ Total turns: {len(transcript)}")

    print("\n── TRANSCRIPT PREVIEW ──────────────────────")
    for turn in transcript[:5]:
        print(f"[{turn['start']}s → {turn['end']}s] "
              f"{turn['speaker']}: {turn['text']}")
    if len(transcript) > 5:
        print(f"... and {len(transcript) - 5} more turns")
    print("────────────────────────────────────────────")

    return result
