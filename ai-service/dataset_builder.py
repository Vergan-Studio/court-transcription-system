import os
import io
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login
from google.colab import userdata

def build_stratified_hf_datasets():
    # 1. Force Login Programmatically
    try:
        token = userdata.get('HF_TOKEN')
        login(token=token, add_to_git_credential=False)
        print("✅ Successfully authenticated with Hugging Face Hub.")
    except Exception as e:
        print(f"⚠️ Warning: Could not login via secrets: {e}")

    unified_list = []
    
    # ... [Keep your Layer 1 and 2 code here] ...

    print("\n📦 Processing Layer 3: Streaming Common Voice...")
    try:
        # Use the token implicitly through the logged-in session
        cv_stream = load_dataset(
            "parquet",
            data_files="hf://datasets/fsicoli/common_voice_19_0@refs/convert/parquet/en/train/*.parquet",
            streaming=True
        )
        
        # Force iteration to verify we have content
        dataset = cv_stream['train'] if isinstance(cv_stream, dict) else cv_stream
        
        samples_collected = 0
        for sample in dataset:
            # Add your filtering/accent logic here
            # Ensure the sample has the keys you expect
            if samples_collected > 2000: break # Safety cap
            
            # ... (your logic) ...
            samples_collected += 1
            
        if samples_collected == 0:
            print("❌ Layer 3 gathered 0 samples. Returning empty structure.")
            return None # This is why your map() was failing!
            
    except Exception as e:
        print(f"❌ Error in Layer 3: {e}")
        return None

    # Only create DatasetDict if we have data
    if not unified_list:
        return None
        
    return DatasetDict({"train": Dataset.from_list(unified_list)})
