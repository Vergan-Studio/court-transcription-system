
import sys
sys.path.insert(0, "/content/drive/MyDrive/speech-to-text-system/ai-service")

from app.services.finetune_service import run_finetuning

model, processor = run_finetuning()
