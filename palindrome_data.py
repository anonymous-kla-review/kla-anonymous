import argparse
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PalindromeDataset(Dataset):
    def __init__(self, data_path=None, data=None):
        if data_path:
            if data_path.endswith('.pt'):
                self.data = torch.load(data_path)
            elif data_path.endswith('.npz'):
                loaded = np.load(data_path)
                self.data = {
                    'input_ids': torch.from_numpy(loaded['input_ids']),
                    'labels': torch.from_numpy(loaded['labels']),
                    'masks': torch.from_numpy(loaded['masks']),
                }
            else:
                raise ValueError(f"Unsupported file extension: {data_path}")
        elif data is not None:
            self.data = data
        else:
            raise ValueError("Must provide data_path or data")

        self.input_ids = self.data['input_ids']
        self.labels = self.data['labels']
        self.masks = self.data['masks']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx],
            'masks': self.masks[idx],
        }


def _sample_source_tokens(rng, seq_len, vocab_size, sep_token_id):
    if sep_token_id == vocab_size - 1:
        return rng.integers(0, vocab_size - 1, size=seq_len, dtype=np.int32)

    source = rng.integers(0, vocab_size, size=seq_len, dtype=np.int32)
    sep_mask = source == sep_token_id
    if np.any(sep_mask):
        source[sep_mask] = (sep_token_id + 1) % vocab_size
    return source


def generate_palindrome_data(
    num_examples,
    seq_len,
    vocab_size=128,
    sep_token_id=None,
    seed=42,
    predict_first_token=False,
):
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2 for the palindrome task")
    if vocab_size < 3:
        raise ValueError("vocab_size must be >= 3")

    if sep_token_id is None:
        sep_token_id = vocab_size - 1
    if sep_token_id < 0 or sep_token_id >= vocab_size:
        raise ValueError(f"sep_token_id must be in [0, {vocab_size - 1}]")

    rng = np.random.default_rng(seed)

    input_ids_list = []
    labels_list = []
    masks_list = []

    full_len = 2 * seq_len + 1

    desc = f"Gen Palindrome (src_len={seq_len})"
    for _ in tqdm(range(num_examples), desc=desc):
        source = _sample_source_tokens(
            rng=rng,
            seq_len=seq_len,
            vocab_size=vocab_size,
            sep_token_id=sep_token_id,
        )
        target = source[::-1]

        input_ids = np.empty(full_len, dtype=np.int32)
        labels = np.full(full_len, -100, dtype=np.int32)
        masks = np.zeros(full_len, dtype=np.bool_)

        input_ids[:seq_len] = source
        input_ids[seq_len] = sep_token_id
        input_ids[seq_len + 1:] = target

        if predict_first_token:
            labels[seq_len] = target[0]
            masks[seq_len] = True

        # Causal LM objective over the generated suffix.
        labels[seq_len + 1: full_len - 1] = target[1:]
        masks[seq_len + 1: full_len - 1] = True

        input_ids_list.append(input_ids.astype(np.int64))
        labels_list.append(labels.astype(np.int64))
        masks_list.append(masks.astype(bool))

    return {
        'input_ids': np.stack(input_ids_list),
        'labels': np.stack(labels_list),
        'masks': np.stack(masks_list),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_train", type=int, default=20000)
    parser.add_argument("--num_val", type=int, default=2000)
    parser.add_argument("--num_test", type=int, default=2000)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--sep_token_id", type=int, default=-1)
    parser.add_argument("--predict_first_token", action="store_true")
    args = parser.parse_args()

    sep_token_id = args.sep_token_id if args.sep_token_id >= 0 else None

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Generating Train ({args.num_train})...")
    train_data = generate_palindrome_data(
        num_examples=args.num_train,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        sep_token_id=sep_token_id,
        seed=args.seed,
        predict_first_token=args.predict_first_token,
    )
    np.savez_compressed(os.path.join(args.save_dir, "train.npz"), **train_data)

    print(f"Generating Val ({args.num_val})...")
    val_data = generate_palindrome_data(
        num_examples=args.num_val,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        sep_token_id=sep_token_id,
        seed=args.seed + 1,
        predict_first_token=args.predict_first_token,
    )
    np.savez_compressed(os.path.join(args.save_dir, "val.npz"), **val_data)

    print(f"Generating Test ({args.num_test})...")
    test_data = generate_palindrome_data(
        num_examples=args.num_test,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        sep_token_id=sep_token_id,
        seed=args.seed + 2,
        predict_first_token=args.predict_first_token,
    )
    np.savez_compressed(os.path.join(args.save_dir, "test.npz"), **test_data)

    print("Done.")
