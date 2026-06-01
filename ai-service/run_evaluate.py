import sys
sys.path.insert(0, "/content/drive/MyDrive/speech-to-text-system/ai-service")

from app.services.evaluate_service import run_evaluation
from app.config import RAW_TEST_CSV


def evaluate_model():
    return run_evaluation(
        test_csv=RAW_TEST_CSV,
        label="whisper_african_v2"
    )


if __name__ == "__main__":
    evaluate_model()