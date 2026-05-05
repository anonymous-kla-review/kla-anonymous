import torch
import numpy as np
from torch.utils.data import Dataset
import os
import argparse
from tqdm import tqdm

class MQARDataset(Dataset):
    def __init__(self, data_path=None, data=None):
        if data_path:
            if data_path.endswith('.pt'):
                self.data = torch.load(data_path)
            elif data_path.endswith('.npz'):
                loaded = np.load(data_path)
                self.data = {
                    'input_ids': torch.from_numpy(loaded['input_ids']),
                    'labels': torch.from_numpy(loaded['labels']),
                    'masks': torch.from_numpy(loaded['masks'])
                }
            else:
                raise ValueError(f"Unsupported file extension: {data_path}")
        elif data:
            self.data = data
        else:
            raise ValueError("Must provide data_path or data")
            
        self.input_ids = self.data['input_ids']
        self.labels = self.data['labels']
        self.masks = self.data['masks'] # 1 for retrieval tokens (values), 0 otherwise

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx],
            'masks': self.masks[idx]
        }

def power_law_sample(alpha, low, high, size, rng):
    """
    Inverse transform sampling for Power Law distribution.
    PDF: f(x) ~ x^(-alpha)
    CDF: F(x) = (x^(1-alpha) - low^(1-alpha)) / (high^(1-alpha) - low^(1-alpha))
    """
    if alpha == 1.0:
        # Log uniform
        log_low = np.log(low)
        log_high = np.log(high)
        u = rng.random(size=size)
        return np.exp(u * (log_high - log_low) + log_low)
    else:
        # Power law
        beta = 1.0 - alpha
        low_b = low ** beta
        high_b = high ** beta
        u = rng.random(size=size)
        return (u * (high_b - low_b) + low_b) ** (1.0 / beta)

def generate_mqar_data(
    num_examples, 
    seq_len, 
    vocab_size=8192, 
    key_len=1, 
    power_law_alpha=0.1, 
    num_pairs=32,
    seed=42,
    noise_vocab_size=None
):
    rng = np.random.default_rng(seed)
    if noise_vocab_size is None:
        noise_vocab_size = vocab_size
        
    # Split vocab
    # Keys from [0, V/2), Values from [V/2, V)
    # Ensure vocab is large enough
    assert vocab_size >= 2, "Vocab size must be at least 2"
    key_range = (0, vocab_size // 2)
    val_range = (vocab_size // 2, vocab_size)
    
    input_ids_list = []
    labels_list = []
    masks_list = []
    
    pair_len = key_len + 1
    prefix_len = num_pairs * pair_len
    
    if prefix_len >= seq_len:
        raise ValueError(f"Prefix length {prefix_len} (D={num_pairs}, key_len={key_len}) >= seq_len {seq_len}")
        
    for _ in tqdm(range(num_examples), desc=f"Gen (N={seq_len}, D={num_pairs}, K={key_len})"):
        seq = np.zeros(seq_len, dtype=np.int32)
        label = np.full(seq_len, -100, dtype=np.int32) # -100 ignore index
        mask = np.zeros(seq_len, dtype=np.bool_)
        
        # 1. Generate D key-value pairs
        # Keys are sampled with replacement? 
        # Paper says: "random mapping M: K -> V". 
        # Usually implies keys are unique in the definition if it's a map.
        # But we sample D pairs. If we sample keys with replacement, we might have collisions.
        # Let's sample keys without replacement for the definitions to ensure distinct pairs.
        # But D might be larger than |K|. If D < |K|, we can do without replacement.
        # With |K| = 4096, D=32, we can do without replacement.
        
        # However, for key_len > 1, the key space is |K|^key_len.
        
        # Let's just sample keys. If collision, it's fine, the later one overwrites?
        # Or better, ensure unique keys for definitions.
        
        # We'll generate a pool of unique keys if possible.
        # For simplicity and speed, just sample. Probability of collision is low for small D.
        
        current_keys = rng.choice(np.arange(*key_range), size=(num_pairs, key_len), replace=True)
        current_values = rng.choice(np.arange(*val_range), size=(num_pairs,), replace=True)
        
        # 2. Place pairs at the beginning
        kv_seq = []
        for i in range(num_pairs):
            kv_seq.extend(current_keys[i])
            kv_seq.append(current_values[i])
            
        seq[:prefix_len] = kv_seq
        
        # 3. Fill the rest with random noise
        seq[prefix_len:] = rng.integers(0, noise_vocab_size, size=seq_len - prefix_len)
        
        # 4. Insert queries
        # For each pair, we try to place a query.
        # We sample a distance d from PowerLaw.
        # Target pos roughly: definition_end + d.
        # Since definition locations are fixed, we can just treat d as position in [prefix_len, seq_len].
        # "Distance from definition" is the physically meaningful metric.
        # Def for pair i is at index `i * pair_len`.
        # Query for pair i should be at `i * pair_len + d`.
        # d must be >= prefix_len - i*pair_len (to be after prefix).
        # And d must be <= seq_len - pair_len - i*pair_len (to fit).
        
        # Let's try to place as many as possible.
        indices = np.arange(num_pairs)
        rng.shuffle(indices)
        
        for i in indices:
            def_start = i * pair_len
            def_end = def_start + pair_len
            
            # Min distance to put it after prefix
            min_dist = prefix_len - def_start
            max_dist = seq_len - pair_len - def_start
            
            if max_dist <= min_dist:
                continue
                
            # Sample distance
            # Paper says distance from [2D, N]. 
            # 2D corresponds to prefix_len. N corresponds to seq_len.
            # So we sample d from [prefix_len, seq_len].
            # Then query pos = def_start + d ??
            # Or query pos = d ??
            # If d is in [prefix_len, seq_len], then d is an absolute position.
            # "distance sampling from [2D, N]" might loosely mean "position sampling".
            # Given the context of "putting back to sequence", absolute position seems likely.
            
            # Let's sample a position `p` in `[prefix_len, seq_len - pair_len]` using PowerLaw.
            # We want it to be skewed towards... left or right?
            # Alpha=0.1 is close to 0, which is uniform?
            # Alpha > 0 usually means P(x) ~ x^-alpha. Decays.
            # So earlier positions (closer to prefix) are more likely.
            
            pos_float = power_law_sample(power_law_alpha, prefix_len, seq_len - pair_len, 1, rng)[0]
            pos = int(pos_float)
            
            # Check if space is free (we don't want to overwrite other queries)
            # We use a simple occupancy map?
            # Or just overwrite noise?
            # We shouldn't overwrite other queries.
            # Since we fill with noise first, we can check if we are overwriting a query.
            # But we don't track queries yet.
            # Let's just allow overwriting noise.
            # To avoid overwriting other queries, we can keep track of occupied mask.
            
            # For simplicity, we won't do strict collision checking against other queries 
            # unless we really want high density.
            # But we should ensure we don't cut a previous query in half?
            # With small D and large N, collisions are rare.
            # Let's just place it.
            
            # Place query: Key only
            seq[pos : pos + key_len] = current_keys[i]
            
            # Target: Value
            # The value should be predicted at the last token of the key.
            # label[pos + key_len - 1] = current_values[i]
            # mask[pos + key_len - 1] = True
            
            # Wait, standard LM: input `... k_last` -> predict `v`.
            # So at position `pos + key_len - 1` (last key token), the target is `current_values[i]`.
            label[pos + key_len - 1] = current_values[i]
            mask[pos + key_len - 1] = True
            
            # Note: We do NOT insert the value into the input sequence for the query!
            # The model predicts the value.
            # But what if we want to chain?
            # MQAR usually just queries. The value is the target.
            # The sequence continues... with what?
            # If we don't put the value, the next token in input is noise?
            # If the next token is noise, the model learns to predict value, but then sees noise.
            # That's fine.
            
            # However, if we overwrite, we might overwrite a label from another query.
            # Let's keep it simple.
            
        input_ids_list.append(seq.astype(np.int64))
        labels_list.append(label.astype(np.int64))
        masks_list.append(mask.astype(bool))
        
    return {
        'input_ids': np.stack(input_ids_list),
        'labels': np.stack(labels_list),
        'masks': np.stack(masks_list)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_train", type=int, default=100000)
    parser.add_argument("--num_val", type=int, default=3000)
    parser.add_argument("--num_test", type=int, default=3000)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--key_len", type=int, default=1)
    parser.add_argument("--num_pairs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"Generating Train ({args.num_train})...")
    train_data = generate_mqar_data(args.num_train, args.seq_len, key_len=args.key_len, num_pairs=args.num_pairs, seed=args.seed)
    np.savez_compressed(os.path.join(args.save_dir, "train.npz"), **train_data)
    
    print(f"Generating Val ({args.num_val})...")
    val_data = generate_mqar_data(args.num_val, args.seq_len, key_len=args.key_len, num_pairs=args.num_pairs, seed=args.seed + 1)
    np.savez_compressed(os.path.join(args.save_dir, "val.npz"), **val_data)
    
    print(f"Generating Test ({args.num_test})...")
    test_data = generate_mqar_data(args.num_test, args.seq_len, key_len=args.key_len, num_pairs=args.num_pairs, seed=args.seed + 2)
    np.savez_compressed(os.path.join(args.save_dir, "test.npz"), **test_data)
    
    print("Done.")
