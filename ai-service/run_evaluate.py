
import sys
sys.path.insert(0, "/content/drive/MyDrive/speech-to-text-system/ai-service")

from app.services.evaluate_service import run_evaluation

# Evaluate V1 model with post-processing
summary = run_evaluation(
    test_csv="/content/drive/MyDrive/datasets/afrispeech/metadata/test_processed.csv",
    label="final_evaluation_v1_postprocessed"
)

print("\nFINAL COMPARISON SUMMARY")
print("="*50)
print(f"Baseline (no ETL):                35.22%")
print(f"After ETL:                        25.50%")
print(f"V1 fine-tune:                     18.83%")
print(f"V1 + post-processing:             {summary['average_wer']}%")
print(f"V2 fine-tune (augmented):         20.37%")
print("="*50)
