from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

# ==================================================
# PROJECT ROOT
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ==================================================
# DATASETS
# ==================================================

DATASETS_DIR = PROJECT_ROOT / "datasets"

AFRISPEECH_DIR = DATASETS_DIR / "afrispeech"
FLEURS_DIR     = DATASETS_DIR / "fleurs"
COMBINED_DIR   = DATASETS_DIR / "combined"



# ==================================================
# METADATA
# ==================================================

TRAIN_CSV = COMBINED_DIR / "metadata" / "train.csv"
VAL_CSV   = COMBINED_DIR / "metadata" / "val.csv"
TEST_CSV  = COMBINED_DIR / "metadata" / "test.csv"

# ==================================================
# MODELS
# ==================================================

MODELS_DIR = PROJECT_ROOT / "ai-service" / "models"

BASE_MODEL = "openai/whisper-medium"

MODEL_DIR = MODELS_DIR / "whisper-cameroon-v1"

# ==================================================
# OUTPUTS
# ==================================================

OUTPUTS_DIR = PROJECT_ROOT / "ai-service" / "outputs"

RESULTS_DIR = OUTPUTS_DIR / "evaluations"

# ==================================================
# DEVICE
# ==================================================

DEVICE = "cpu"

# ==================================================
# TOKENS
# ==================================================

HF_TOKEN = os.getenv("HF_TOKEN")
PYANNOTE_TOKEN = os.getenv("PYANNOTE_TOKEN")