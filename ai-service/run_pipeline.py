
import sys
sys.path.insert(0, BASE_DIR)

from google.colab import userdata
from app.services.pipeline_service import run_pipeline

result = run_pipeline(
    audio_path=AUDIO_PATH,
    hf_token=userdata.get("HF_TOKEN"),
    output_dir=OUTPUT_DIR
)

print("\n" + "="*50)
print(f"  DONE — {result['total_turns']} turns transcribed")
print(f"  Output saved to: {OUTPUT_DIR}")
print("="*50)
