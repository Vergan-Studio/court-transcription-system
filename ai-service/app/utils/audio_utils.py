
import os
import subprocess

def preprocess_audio(input_path: str, output_path: str) -> str:
    """
    Convert any audio file to mono, 16kHz, wav format.
    This is the standard format expected by both Whisper and Pyannote.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    command = [
        'ffmpeg',
        '-i', input_path,        # input file
        '-ac', '1',              # mono
        '-ar', '16000',          # 16kHz sample rate
        '-acodec', 'pcm_s16le',  # standard wav encoding
        '-y',                    # overwrite output if exists
        output_path
    ]
    
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    if result.returncode != 0:
        raise RuntimeError(
            f"Audio preprocessing failed:\n{result.stderr.decode()}"
        )
    
    print(f"✅ Audio preprocessed: {output_path}")
    return output_path


def validate_audio(file_path: str) -> bool:
    """
    Check that an audio file exists and is readable.
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    if os.path.getsize(file_path) == 0:
        print(f"❌ File is empty: {file_path}")
        return False
    
    print(f"✅ Audio file valid: {file_path}")
    return True
