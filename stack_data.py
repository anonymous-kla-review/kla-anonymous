import argparse
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class StackDataset(Dataset):
    def __init__(self, data_path=None, data=None):
        if data_path:
            if data_path.endswith(".pt"):
                self.data = torch.load(data_path)
            elif data_path.endswith(".npz"):
                loaded = np.load(data_path)
                self.data = {
                    "input_ids": torch.from_numpy(loaded["input_ids"]),
                    "labels": torch.from_numpy(loaded["labels"]),
                    "masks": torch.from_numpy(loaded["masks"]),
                }
            else:
                raise ValueError(f"Unsupported file extension: {data_path}")
        elif data is not None:
            self.data = data
        else:
            raise ValueError("Must provide data_path or data")

        self.input_ids = self.data["input_ids"]
        self.labels = self.data["labels"]
        self.masks = self.data["masks"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "masks": self.masks[idx],
        }


def generate_stack_data(
    num_examples,
    seq_len,
    vocab_size=128,
    num_stacks=64,
    num_values=26,
    push_prob=0.6,
    seed=42,
):
    if seq_len < 3:
        raise ValueError("seq_len must be >= 3 for stack task")
    if num_stacks < 1:
        raise ValueError("num_stacks must be >= 1")
    if num_values < 2:
        raise ValueError("num_values must be >= 2")
    if not 0.0 < push_prob < 1.0:
        raise ValueError("push_prob must be in (0, 1)")

    value_start = num_stacks
    value_end = value_start + num_values
    push_token = value_end
    pop_token = push_token + 1
    query_token = pop_token + 1
    noise_start = query_token + 1

    min_vocab = noise_start
    if vocab_size < min_vocab:
        raise ValueError(
            f"vocab_size={vocab_size} is too small. Need at least {min_vocab} for "
            f"num_stacks={num_stacks}, num_values={num_values}."
        )

    rng = np.random.default_rng(seed)

    input_ids_list = []
    labels_list = []
    masks_list = []

    desc = f"Gen Stack (len={seq_len}, stacks={num_stacks})"
    for _ in tqdm(range(num_examples), desc=desc):
        tokens = np.empty(seq_len, dtype=np.int32)
        labels = np.full(seq_len, -100, dtype=np.int32)
        masks = np.zeros(seq_len, dtype=np.bool_)

        states = [[] for _ in range(num_stacks)]

        cursor = 0
        while cursor + 3 <= seq_len:
            non_empty_ids = [sid for sid in range(num_stacks) if states[sid]]
            can_pop = len(non_empty_ids) > 0
            do_push = (not can_pop) or (rng.random() < push_prob)

            if do_push:
                sid = int(rng.integers(0, num_stacks))
                value = int(rng.integers(value_start, value_end))

                tokens[cursor] = push_token
                tokens[cursor + 1] = sid
                tokens[cursor + 2] = value

                states[sid].append(value)
            else:
                sid = int(rng.choice(non_empty_ids))
                target = states[sid].pop()

                tokens[cursor] = pop_token
                tokens[cursor + 1] = sid
                tokens[cursor + 2] = query_token

                # Predict the popped value right after reading <pop, stack_id>.
                labels[cursor + 1] = target
                masks[cursor + 1] = True

            cursor += 3

        if cursor < seq_len:
            tail_len = seq_len - cursor
            if noise_start < vocab_size:
                tokens[cursor:] = rng.integers(noise_start, vocab_size, size=tail_len)
            else:
                tokens[cursor:] = query_token

        input_ids_list.append(tokens.astype(np.int64))
        labels_list.append(labels.astype(np.int64))
        masks_list.append(masks.astype(bool))

    return {
        "input_ids": np.stack(input_ids_list),
        "labels": np.stack(labels_list),
        "masks": np.stack(masks_list),
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
    parser.add_argument("--num_stacks", type=int, default=64)
    parser.add_argument("--num_values", type=int, default=26)
    parser.add_argument("--push_prob", type=float, default=0.6)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Generating Train ({args.num_train})...")
    train_data = generate_stack_data(
        num_examples=args.num_train,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        num_stacks=args.num_stacks,
        num_values=args.num_values,
        push_prob=args.push_prob,
        seed=args.seed,
    )
    np.savez_compressed(os.path.join(args.save_dir, "train.npz"), **train_data)

    print(f"Generating Val ({args.num_val})...")
    val_data = generate_stack_data(
        num_examples=args.num_val,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        num_stacks=args.num_stacks,
        num_values=args.num_values,
        push_prob=args.push_prob,
        seed=args.seed + 1,
    )
    np.savez_compressed(os.path.join(args.save_dir, "val.npz"), **val_data)

    print(f"Generating Test ({args.num_test})...")
    test_data = generate_stack_data(
        num_examples=args.num_test,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        num_stacks=args.num_stacks,
        num_values=args.num_values,
        push_prob=args.push_prob,
        seed=args.seed + 2,
    )
    np.savez_compressed(os.path.join(args.save_dir, "test.npz"), **test_data)

    print("Done.")
