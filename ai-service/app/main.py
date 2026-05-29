from fastapi import FastAPI, UploadFile, File
from app.services.pipeline_service import process_audio
from app.utils.audio_service import save_upload, convert_to_wav, cleanup

app = FastAPI()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):

    # 1. Save upload safely
    input_path = await save_upload(file)

    try:
        # 2. Convert audio → WAV
        wav_path = convert_to_wav(input_path)

        # 3. Run AI pipeline
        result = process_audio(wav_path)

        return {
            "status": "success",
            "transcript": result
        }

    finally:
        # 4. Always clean temp files
        cleanup(input_path, wav_path)