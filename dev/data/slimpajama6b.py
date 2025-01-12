import os
import argparse
import multiprocessing as mp

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoTokenizer
from data_common import write_datafile

# ------------------------------------------

parser = argparse.ArgumentParser(description="SlimPajama-6B dataset preprocessing")
parser.add_argument("-v", "--version", type=str, default="6B", help="SlimPajama data sample size, 6B")
parser.add_argument("-m", "--model_desc", type=str, default="gpt-2", help="Model descriptor, gpt-2|llama-3")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each data shard in the output .bin files, in tokens")
args = parser.parse_args()

# SlimPajama has the 6B version available
assert args.version == "6B", "version must be 6B"
local_dir = "slimpajama6B"
remote_name = "sample-6B"

# Create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Download the dataset
slimpajama = load_dataset("DKYoon/SlimPajama-6B", name=remote_name, split="train")
name = "slimpajama"

def tokenize_llama(doc):
    # Tokenizes a single document and returns a numpy array of uint32 tokens
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False, verbose=False, split_special_tokens=True)
    eot = tokenizer.encode('')[0]  # By default the tokenizer adds the EOT token (128000)
    tokens = [eot]  # The special <|endoftext|> token delimits all documents
    tokens.extend(encode(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**32).all(), "Token dictionary too large for uint32"
    tokens_np_uint = tokens_np.astype(np.uint32)
    return tokens_np_uint

def tokenize_gpt2(doc):
    # Tokenizes a single document and returns a numpy array of uint16 tokens
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode_ordinary(s)
    eot = enc._special_tokens['<|endoftext|>']  # End of text token
    tokens = [eot]  # The special <|endoftext|> token delimits all documents
    tokens.extend(encode(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token dictionary too large for uint16"
    tokens_np_uint = tokens_np.astype(np.uint16)
    return tokens_np_uint

token_dtype = {
    "gpt-2": np.uint16,
    "llama-3": np.uint32
}[args.model_desc]

# Tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count() - 2)  # Don't hog the entire system
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # Preallocate buffer to hold current shard
    all_tokens_np = np.empty((args.shard_size,), dtype=token_dtype)
    token_count = 0
    progress_bar = None

    tokenize = lambda x: None
    if args.model_desc == "gpt-2":
        tokenize = tokenize_gpt2
    elif args.model_desc == "llama-3":
        tokenize = tokenize_llama
    else:
        raise ValueError(f"Unknown model {args.model_desc}")

    for tokens in pool.imap(tokenize, slimpajama, chunksize=16):

        # Is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < args.shard_size:
            # Simply append tokens to current shard
            all_tokens_np[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)
            # Update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # Write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
            # Split the document into whatever fits in this shard; the remainder goes to next one
            remainder = args.shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np.tolist(), args.model_desc)
            shard_index += 1
            progress_bar = None
            # Populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    # Write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, (all_tokens_np[:token_count]).tolist(), args.model_desc)
