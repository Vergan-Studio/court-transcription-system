!pip install jiwer -q
import pandas as pd
import whisper
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip, wer
from tqdm import tqdm
import json

# load dataset
df = pd.read_csv(
    "/content/drive/MyDrive/fyp-phase0/phase0_dataset.csv"
)

print("Samples:", len(df))

# load whisper model
model = whisper.load_model("medium.en")

references = []
predictions = []
results = []

# transcribe all 224 samples
for _, row in tqdm(df.iterrows(), total=len(df)):

    try:
        result = model.transcribe(
            row["audio_path"],
            language="en"
        )

        prediction = result["text"].strip()

    except Exception as e:
        print("Error:", row["audio_path"])
        print(e)

        prediction = ""

    references.append(row["transcript"])
    predictions.append(prediction)

    results.append({
        "audio_path": row["audio_path"],
        "reference": row["transcript"],
        "prediction": prediction,
        "speaker_id": row["speaker_id"],
        "duration": row["duration"]
    })

# normalize text before evaluation
transform = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    Strip()
])

references_clean = [transform(r) for r in references]
predictions_clean = [transform(p) for p in predictions]

# compute wer
baseline_wer = wer(
    references_clean,
    predictions_clean
)

print("Baseline WER:", baseline_wer)
print(f"Baseline WER: {baseline_wer*100:.2f}%")

# save predictions
results_df = pd.DataFrame(results)

results_df.to_csv(
    "/content/drive/MyDrive/fyp-phase0/results/baseline_predictions.csv",
    index=False
)

# save metrics
metrics = {
    "model": "whisper-medium.en",
    "samples": len(df),
    "wer": baseline_wer
}

with open(
    "/content/drive/MyDrive/fyp-phase0/results/baseline_metrics.json",
    "w"
) as f:
    json.dump(metrics, f, indent=4)

# Error Analysis
results_df["match"] = (
    results_df["reference"].str.lower()
    ==
    results_df["prediction"].str.lower()
)

print(results_df["match"].value_counts())

# inspect worst case scenarious
results_df[
    results_df["reference"].str.lower()
    !=
    results_df["prediction"].str.lower()
].head(20)
