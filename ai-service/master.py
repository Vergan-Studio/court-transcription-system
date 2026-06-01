#!/usr/bin/env python3
from app.services.dataset_downloader import run_dataset_download
BASE_DIR = "/content/drive/MyDrive/speech-to-text-system/ai-service"


# # ── SETUP FUNCTION ─────────────────────────────
# def setup_environment():
#     try:
#         import torch
#         import transformers
#         import peft
#         print("✅ Environment already ready")

#     except Exception:
#         print("❌ Environment not ready.")
#         print("Run setup_colab.py first.")
#         exit()


# ── PIPELINE FUNCTIONS ─────────────────────────
def run_dataset_pipeline():
    run_dataset_download()

def run_etl():
    from run_etl import main
    main()


def run_train():
    from train import run_finetuning
    run_finetuning()


def run_transcribe():
    audio_path = input("Enter audio path: ").strip()
    from run_pipeline import run_pipeline
    run_pipeline(audio_path)


def run_vocab():
    from run_vocabulary import build_vocab
    build_vocab()


def run_evaluate():
    from run_evaluate import evaluate_model
    evaluate_model()


# ── MENU ───────────────────────────────────────
def menu():
    # setup_environment()  # 👈 IMPORTANT: always run first

    while True:
        print("\n===== MASTER CONTROL =====")
        print("0. Dataset Pipeline")
        print("1. ETL")
        print("2. Train")
        print("3. Transcribe")
        print("4. Vocabulary")
        print("5. Evaluate")
        print("6. Exit")
        print("==========================")

        choice = input("Select option: ").strip()

        if choice == "0":
            run_dataset_pipeline()
        elif choice == "1":
            run_etl()

        elif choice == "2":
            run_train()

        elif choice == "3":
            run_transcribe()

        elif choice == "4":
            run_vocab()

        elif choice == "5":
            run_evaluate()

        elif choice == "6":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    menu()
