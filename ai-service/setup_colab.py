
# ============================================
# RUN THIS AT THE START OF EVERY COLAB SESSION
# ============================================

# Step 1 - Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2 - Install in correct order using shell commands
import subprocess

def pip(*args):
    subprocess.run(['pip', 'install', *args, '-q'], check=True)

# Install pyannote without touching torch
pip('pyannote.audio==4.0.4', '--no-deps')

# Install all pyannote dependencies
pip(
    'asteroid-filterbanks>=0.4',
    'einops>=0.8.1',
    'pyannote-core>=6.0.1',
    'pyannote-database>=6.1.1',
    'pyannote-metrics>=4.0.0',
    'pyannote-pipeline>=4.0.0',
    'pytorch-metric-learning>=2.8.1',
    'torch-audiomentations>=0.12.0',
    'lightning>=2.4',
    'semver>=0.3.0',
    'rich>=13.9.4',
    'huggingface_hub>=0.28.1',
    'safetensors>=0.5.2',
    'sortedcontainers',
    'pytorch-lightning',
    'lightning-utilities',
    'torchmetrics'
)

# Install remaining dependencies
pip(
    'opentelemetry-exporter-otlp>=1.34.0',
    'pyannoteai-sdk>=0.3.0',
    'openai-whisper'
)

# Step 3 - Verify
import torch
import whisper
from pyannote.audio import Pipeline

print("=" * 40)
print("ENVIRONMENT CHECK")
print("=" * 40)
print(f"torch:    {torch.__version__}")
print(f"CUDA:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:      {torch.cuda.get_device_name(0)}")
else:
    print(f"GPU:      not available (CPU mode)")
print(f"whisper:  loaded")
print(f"pyannote: loaded")
print("=" * 40)
print("All systems ready")
