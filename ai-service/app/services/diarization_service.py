
import torch
from pyannote.audio import Pipeline


def load_diarization_pipeline(hf_token: str):
    """
    Load the pyannote speaker diarization pipeline.
    """
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    
    print(f"✅ Diarization pipeline loaded on {device}")
    return pipeline


def run_diarization(pipeline, audio_path: str) -> list:
    """
    Run speaker diarization on an audio file.
    Returns a list of segments with speaker labels and timestamps.
    """
    print(f"🔄 Running diarization on: {audio_path}")
    
    output = pipeline(audio_path)
    
    segments = []

    # pyannote 4.x returns a DiarizeOutput object
    # iterate over speaker_diarization attribute
    for turn, speaker in output.speaker_diarization:
        segments.append({
            "speaker": speaker,
            "start":   round(turn.start, 3),
            "end":     round(turn.end, 3)
        })
    
    print(f"✅ Diarization complete — {len(segments)} segments found")
    return segments
