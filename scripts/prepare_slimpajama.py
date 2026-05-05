import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Set HF Mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Add project root to path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

# Use standalone builder to avoid torch import issues
sys.path.append(str(wd / "GatedDeltaNet" / "scripts"))
try:
    from standalone_builder import PackedDatasetBuilder
except ImportError:
    # Try local import if script is run from scripts dir
    sys.path.append(str(Path(__file__).parent))
    from standalone_builder import PackedDatasetBuilder

from transformers import AutoTokenizer
from datasets import load_dataset

# Configuration
DATASET_NAME = "DKYoon/SlimPajama-6B"
TOKENIZER_NAME = "TinyLlama/TinyLlama_v1.1"
DEFAULT_OUTPUT_DIR = os.environ.get("DATA_DIR", str(wd / "mydata" / "slimpajama"))
CHUNK_SIZE = 1_048_576 # 1M tokens per chunk
DEFAULT_TARGET_TOKENS = 10_000_000_000
DEFAULT_VAL_SAMPLES = 1000
RESUME_STATE_FILE = ".prepare_slimpajama_resume_state.json"

def _chunk_files(directory, prefix):
    files = sorted(directory.glob(f"{prefix}_*.bin"))
    if not files:
        return []

    indices = [int(path.stem.rsplit("_", 1)[-1]) for path in files]
    expected = list(range(len(indices)))
    if indices != expected:
        raise RuntimeError(
            f"Found non-contiguous chunk indices in {directory}: {indices[:5]}..."
        )
    return files


def _backup_path(path):
    candidate = path.with_name(f"{path.name}.resume_backup")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.resume_backup.{suffix}")
        suffix += 1
    return candidate


def _archive_last_chunk(chunk_files):
    if not chunk_files:
        return None

    last_chunk = chunk_files[-1]
    backup = _backup_path(last_chunk)
    last_chunk.rename(backup)
    return backup


def _tokenize_text(tokenizer, text, sep_id, dtype):
    ids = tokenizer.encode(text)
    ids.append(sep_id)
    return np.array(ids, dtype=dtype)


def _flush_builder(builder):
    if builder is not None and getattr(builder, "_idx", 0) > 0:
        builder.write_reminder()


def _load_resume_state(output_dir):
    state_path = Path(output_dir) / RESUME_STATE_FILE
    if not state_path.exists():
        return state_path, None

    with state_path.open("r", encoding="utf-8") as f:
        return state_path, json.load(f)


def _save_resume_state(state_path, state):
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def prepare_data(limit=None, target_tokens=DEFAULT_TARGET_TOKENS):
    print(f"Preparing dataset {DATASET_NAME}...")
    print(f"Output directory: {DEFAULT_OUTPUT_DIR}")
    if limit:
        print(f"Limit: {limit} samples")
    print(f"Target train tokens: {target_tokens:,}")

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

    state_path, resume_state = _load_resume_state(DEFAULT_OUTPUT_DIR)
    train_files = _chunk_files(train_dir, "train_slim")
    val_files = _chunk_files(val_dir, "validation")
    preserved_train_chunks = 0
    preserved_train_tokens = 0
    train_start_index = 0

    if train_files:
        print(f"Found {len(train_files)} existing training chunks.")
        can_reuse_resume_boundary = (
            resume_state is not None
            and resume_state.get("resume_chunks") == len(train_files)
            and Path(resume_state.get("archived_chunk", "")).exists()
        )

        if can_reuse_resume_boundary:
            preserved_train_chunks = len(train_files)
            preserved_train_tokens = preserved_train_chunks * CHUNK_SIZE
            train_start_index = preserved_train_chunks
            print(f"Reusing existing resume boundary from {state_path}")
            print(f"Safe resume point: {preserved_train_tokens:,} train tokens")
        else:
            preserved_train_chunks = max(len(train_files) - 1, 0)
            preserved_train_tokens = preserved_train_chunks * CHUNK_SIZE
            train_start_index = preserved_train_chunks
            archived_chunk = _archive_last_chunk(train_files)
            _save_resume_state(
                state_path,
                {
                    "archived_chunk": str(archived_chunk),
                    "resume_chunks": preserved_train_chunks,
                    "resume_tokens": preserved_train_tokens,
                },
            )
            print(f"Archived trailing training chunk: {archived_chunk}")
            print(f"Saved resume state to: {state_path}")
            print(f"Safe resume point: {preserved_train_tokens:,} train tokens")
    else:
        print("No existing training chunks found. Starting fresh.")

    if preserved_train_tokens >= target_tokens:
        print("Existing training data already meets or exceeds the target token count.")
        return

    # Load Dataset (Streaming)
    print("Loading dataset stream...")
    try:
        print("Loading train split...")
        train_ds = load_dataset(DATASET_NAME, split="train", streaming=True)
        print("Train split loaded.")
    except Exception as e:
        print(f"Error loading train split: {e}")
        return

    # Builders
    print("Initializing builders...")
    train_builder = PackedDatasetBuilder(
        str(train_dir),
        "train_slim",
        CHUNK_SIZE,
        sep_id,
        dtype="auto",
        vocab_size=vocab_size,
        start_index=train_start_index,
    )
    val_builder = None
    has_val_split = False

    if val_files:
        print(f"Keeping existing validation data: {len(val_files)} chunks")
    else:
        try:
            print("Loading validation split...")
            val_ds = load_dataset(DATASET_NAME, split="validation", streaming=True)
            print("Validation split loaded.")
            has_val_split = True
            val_builder = PackedDatasetBuilder(
                str(val_dir),
                "validation",
                CHUNK_SIZE,
                sep_id,
                dtype="auto",
                vocab_size=vocab_size,
            )
        except Exception:
            print("No validation split found. Reusing train split for validation is only supported on a fresh run.")
            if preserved_train_tokens > 0:
                raise RuntimeError(
                    "Cannot safely resume training data without an existing validation set or a dedicated validation split."
                )
            val_builder = PackedDatasetBuilder(
                str(val_dir),
                "validation",
                CHUNK_SIZE,
                sep_id,
                dtype="auto",
                vocab_size=vocab_size,
            )

    print("Processing...")

    if has_val_split:
        print("Processing validation split...")
        val_count = 0
        for item in tqdm(val_ds):
            text = item.get("text", "")
            if not text:
                continue

            arr = _tokenize_text(tokenizer, text, sep_id, train_builder.dtype)
            val_builder.add_array(arr)
            val_count += 1
            if val_count >= DEFAULT_VAL_SAMPLES:
                break

    written_train_tokens = preserved_train_tokens
    skip_train_tokens = preserved_train_tokens
    processed_train_docs = 0
    val_samples_needed = DEFAULT_VAL_SAMPLES if (not has_val_split and not val_files) else 0
    progress = tqdm(total=target_tokens, initial=preserved_train_tokens, unit="tok", unit_scale=True)

    for item in train_ds:
        text = item.get("text", "")
        if not text:
            continue

        arr = _tokenize_text(tokenizer, text, sep_id, train_builder.dtype)

        if val_samples_needed > 0:
            val_builder.add_array(arr)
            val_samples_needed -= 1
            continue

        if skip_train_tokens >= arr.shape[0]:
            skip_train_tokens -= arr.shape[0]
            continue

        if skip_train_tokens > 0:
            arr = arr[skip_train_tokens:]
            skip_train_tokens = 0

        remaining = target_tokens - written_train_tokens
        if remaining <= 0:
            break

        if arr.shape[0] > remaining:
            arr = arr[:remaining]

        train_builder.add_array(arr)
        written_train_tokens += arr.shape[0]
        processed_train_docs += 1
        progress.update(arr.shape[0])

        if limit and processed_train_docs >= limit:
            print(f"Reached limit of {limit} resumed training samples.")
            break

        if written_train_tokens >= target_tokens:
            break

    progress.close()

    _flush_builder(train_builder)
    _flush_builder(val_builder)

    print(f"Done! Data prepared in {DEFAULT_OUTPUT_DIR}")
    print(f"Train tokens available: {written_train_tokens:,}")
    if written_train_tokens < target_tokens:
        print(f"Dataset exhausted before reaching target. Missing {target_tokens - written_train_tokens:,} tokens.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SlimPajama dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=DEFAULT_TARGET_TOKENS,
        help="Target number of training tokens to materialize",
    )
    args = parser.parse_args()

    prepare_data(limit=args.limit, target_tokens=args.target_tokens)
