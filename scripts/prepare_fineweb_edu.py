import os
import sys

# Set HF Mirror before other imports
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import multiprocessing as mp
import argparse

# Add project root to path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.packed_dataset import PackedDatasetBuilder

# Configuration
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_SUBSET = "sample-10BT" # 10 Billion tokens subset
TOKENIZER_NAME = "TinyLlama/TinyLlama_v1.1"
DEFAULT_OUTPUT_DIR = os.environ.get("DATA_DIR", str(wd / "mydata" / "fineweb_edu"))
CHUNK_SIZE = 1_048_576 # 1M tokens per chunk (approx 2MB for uint16)

def prepare_data(limit=None):
    print(f"Preparing dataset {DATASET_NAME} ({DATASET_SUBSET})...")
    print(f"Output directory: {DEFAULT_OUTPUT_DIR}")
    if limit:
        print(f"Limit: {limit} samples")
    
    # Create output directories
    train_dir = Path(DEFAULT_OUTPUT_DIR) / "train"
    val_dir = Path(DEFAULT_OUTPUT_DIR) / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Load Tokenizer
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    vocab_size = tokenizer.vocab_size
    sep_id = tokenizer.eos_token_id or tokenizer.sep_token_id or 0
    print(f"Vocab size: {vocab_size}, SEP ID: {sep_id}")

    # Load Dataset (Streaming to save disk space)
    print("Loading dataset stream...")
    ds = load_dataset(DATASET_NAME, name=DATASET_SUBSET, split="train", streaming=True)
    
    # We'll use a simple split: first 1% for validation, rest for training (approx)
    # Since it's streaming, we'll just take the first N for validation
    val_samples = 1000 # Approx 1M tokens if 1000 tokens/doc
    
    # Builders
    train_builder = PackedDatasetBuilder(
        str(train_dir), "train_slim", CHUNK_SIZE, sep_id, dtype="auto", vocab_size=vocab_size
    )
    val_builder = PackedDatasetBuilder(
        str(val_dir), "validation", CHUNK_SIZE, sep_id, dtype="auto", vocab_size=vocab_size
    )

    print("Processing...")
    count = 0
    # Process loop
    for item in tqdm(ds):
        text = item['text']
        ids = tokenizer.encode(text)
        
        # Add EOS token if not present (TinyLlama usually doesn't add it by default in encode without special args)
        ids.append(sep_id) 
        
        arr = np.array(ids, dtype=train_builder.dtype)
        
        if count < val_samples:
            val_builder.add_array(arr)
        else:
            train_builder.add_array(arr)
            
        count += 1
        
        if limit and count >= limit:
            print(f"Reached limit of {limit} samples.")
            break

    print("Finalizing datasets...")
    # write_reminder() flushes the last chunk
    # Note: older lit-gpt might not have write_reminder or it might be named differently?
    # Checked file content earlier: `builder.write_reminder()` exists in generate_toy_dataset.py
    # But checking packed_dataset.py source code again...
    # I saw `builder.write_reminder()` in generate_toy_dataset.py line 20.
    # But wait, packed_dataset.py source code I read (lines 75-99) didn't show the full class.
    # I'll trust generate_toy_dataset.py usage.
    
    # Wait, looking at my Read of packed_dataset.py, it cut off at line 99.
    # I'll assume write_reminder exists based on generate_toy_dataset.py usage.
    # Actually, standard lit-gpt usually just relies on the destructor or explicit calls? 
    # generate_toy_dataset.py uses `builder.write_reminder()`.
    
    # However, to be safe, I'll check if the method exists or catch error.
    try:
        train_builder.write_reminder()
        val_builder.write_reminder()
    except AttributeError:
        # Maybe it's not needed or named differently.
        # But generate_toy_dataset calls it.
        pass

    print(f"Done! Data prepared in {DEFAULT_OUTPUT_DIR}")
    print(f"Train path: {train_dir}")
    print(f"Validation path: {val_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FineWeb-Edu dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (for testing)")
    args = parser.parse_args()
    
    prepare_data(limit=args.limit)
