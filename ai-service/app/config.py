import os

# ── MODEL CONFIG ─────────────────────────────────
BASE_MODEL = "openai/whisper-medium.en"

# OUTPUT DIR = where checkpoints + final model go
OUTPUT_DIR = "/content/drive/MyDrive/speech-to-text-system/ai-service/models/whisper-african-v2"

# optional alias (same path used for evaluation or loading)
MODEL_DIR = OUTPUT_DIR  # ✅ CHANGED: removed duplication confusion


# ── DATA ─────────────────────────────────────────
TRAIN_CSV = "/content/drive/MyDrive/datasets/afrispeech/metadata/train_split.csv"
VAL_CSV   = "/content/drive/MyDrive/datasets/afrispeech/metadata/val_split.csv"

RAW_DIR = "/content/drive/MyDrive/datasets/afrispeech"
PROCESSED_DIR = f"{RAW_DIR}/processed"
METADATA_DIR  = f"{RAW_DIR}/metadata"

RAW_TRAIN_CSV = f"{RAW_DIR}/transcripts/train.csv"
RAW_TEST_CSV  = f"{RAW_DIR}/transcripts/test.csv"

EXTERNAL_DATASETS_DIR = "/content/external_datasets"
CACHE_DIR = "/content/cache"


# ── TRAINING HYPERPARAMETERS ─────────────────────
MAX_STEPS     = 1000
SAVE_STEPS    = 200
EVAL_STEPS    = 200
BATCH_SIZE    = 4
LEARNING_RATE = 1e-4
SAMPLE_RATE   = 16000

MIN_DURATION  = 1.0
MAX_DURATION  = 30.0
TARGET_DB     = -20.0


# ── RESUME TRAINING (IMPORTANT FOR COLAB) ────────
RESUME_FROM_CHECKPOINT = True
CHECKPOINT_DIR = OUTPUT_DIR  # HuggingFace trainer expects same folder


# ── PIPELINE ──────────────────────────────────────
MIN_SEGMENT_DURATION = 1.0


# ── EVALUATION ────────────────────────────────────
RESULTS_DIR = "/content/drive/MyDrive/speech-to-text-system/ai-service/outputs/evaluations"


# ── VOCABULARY ────────────────────────────────────
VOCAB_PATH = "/content/drive/MyDrive/speech-to-text-system/ai-service/app/utils/court_vocabulary.json"


# ── MODEL OPTIMIZATION FLAGS ──────────────────────
USE_FP16 = True
USE_GRADIENT_CHECKPOINTING = True
NUM_WORKERS = 2


# ── COURT VOCABULARY ──────────────────────────────
COURT_VOCABULARY = {
    "legal_terms": [
        "plaintiff", "defendant", "prosecution", "defence",
        "affidavit", "subpoena", "habeas corpus", "voir dire",
        "arraignment", "indictment", "acquittal", "adjournment",
        "bailiff", "barrister", "solicitor", "magistrate",
        "jurisdiction", "admissible", "inadmissible", "hearsay",
        "cross examination", "examination in chief", "deposition",
        "exhibit", "testimony", "perjury", "contempt of court",
        "injunction", "mandate", "verdict", "sentence",
        "pleading", "motion", "objection", "sustained", "overruled"
    ],
    "african_legal_terms": [
        "customary law", "native court", "district court",
        "high court", "court of appeal", "supreme court",
        "magistrate court", "tribunal", "arbitration",
        "land dispute", "chieftaincy", "traditional ruler"
    ],
    "cameroon_specific": [
        "OHADA", "CEMAC", "anglophone", "francophone",
        "common law", "civil law", "divisional officer",
        "senior divisional officer", "governor", "prefect"
    ],
    "medical_legal": [
        "forensic", "autopsy", "toxicology", "pathology",
        "DNA evidence", "ballistics", "fingerprints",
        "chain of custody", "expert witness"
    ]
}


SAVE_PATH = VOCAB_PATH


# ── TEXT NORMALIZATION ────────────────────────────
PUNCTUATION_RULES = {
    r"\bfull stop\b": ".",
    r"\bnext line\b": "\n",
    r"\bcomma\b": ",",
    r"\bquestion mark\b": "?",
    r"\bexclamation mark\b": "!",
    r"\bopen bracket\b": "(",
    r"\bclose bracket\b": ")",
    r"\bopen quote\b": "\"",
    r"\bclose quote\b": "\"",
}

MEDICAL_FIXES = {
    r"\bcircles per minute\b": "cycles/min",
    r"\bcyclesmin\b": "cycles/min",
    r"\bafebri\b": "afebrile",
    r"\banictary\b": "anicteric",
    r"\bacian nose\b": "acyanosed",
    r"\bcrass crepitations\b": "coarse crepitations",
}
