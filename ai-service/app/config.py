from pathlib import Path
import torch
from google.colab import userdata

# ==================================================
# PROJECT ROOT
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ==================================================
# DATASETS & METADATA
# ==================================================

DATASETS_DIR = PROJECT_ROOT / "datasets"

AFRISPEECH_DIR = DATASETS_DIR / "afrispeech"
FLEURS_DIR     = DATASETS_DIR / "fleurs"
COMBINED_DIR   = DATASETS_DIR / "combined"

TRAIN_CSV = COMBINED_DIR / "metadata" / "train.csv"
VAL_CSV   = COMBINED_DIR / "metadata" / "val.csv"
TEST_CSV  = COMBINED_DIR / "metadata" / "test.csv"

# ==================================================
# MODELS & OUTPUTS
# ==================================================

MODELS_DIR = PROJECT_ROOT / "ai-service" / "models"
BASE_MODEL = "openai/whisper-medium"
MODEL_DIR = MODELS_DIR / "whisper-cameroon-v1"

OUTPUTS_DIR = PROJECT_ROOT / "ai-service" / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "evaluations"

# Map the Trainer directories expected by finetune_service.py
OUTPUT_DIR = str(MODEL_DIR)
CHECKPOINT_DIR = str(MODEL_DIR)
VOCAB_PATH = PROJECT_ROOT / "ai-service" / "court_vocab.json"

# ==================================================
# TRAINING HYPERPARAMETERS (Required by Trainer)
# ==================================================

SAMPLE_RATE = 16000
BATCH_SIZE = 4             # Optimized for Colab GPU VRAM limits (with accumulation)
LEARNING_RATE = 1e-5       # Standard stable rate for Whisper LoRA fine-tuning
MAX_STEPS = 1000           # Total training iterations
EVAL_STEPS = 200           # Evaluate model performance every X steps
SAVE_STEPS = 200           # Save a checkpoint every X steps
RESUME_FROM_CHECKPOINT = False

# ==================================================
# DEVICE (Dynamic GPU Detection)
# ==================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================================================
# SECURE GOOGLE COLAB TOKENS
# ==================================================

try:
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = None

try:
    PYANNOTE_TOKEN = userdata.get("PYANNOTE_TOKEN")
except Exception:
    PYANNOTE_TOKEN = None
