
import sys
sys.path.insert(0, "/content/drive/MyDrive/speech-to-text-system/ai-service")

from app.services.vocabulary_service import run_vocabulary_setup

vocabulary = run_vocabulary_setup()
