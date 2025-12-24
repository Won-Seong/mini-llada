import os
import json
from glob import glob
from itertools import chain
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast

def train_tokenizer(files, vocab_size=52000):
    """
    Train a BPE tokenizer on the provided files.
    """
    print(f"Training tokenizer on {len(files)} files with vocab_size={vocab_size}...")
    
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # And then train
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=[
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|padding|>",
            "<|unk|>",
            "<|mask|>"
        ]
    )
    
    # Use an iterator to yield lines from files to avoid loading everything into memory
    def batch_iterator(batch_size=1000):
        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                data = data['data_info']
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "contents" in item:
                             yield item["contents"]
                elif isinstance(data, dict) and "contents" in data:
                    yield data["contents"]
            except Exception as e:
                print(f"Skipping {file_path} due to error: {e}")
                continue

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    # Wrap in transformers PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
        pad_token="<|padding|>",
        unk_token="<|unk|>", # BPE usually doesn't have UNK, mapping to EOS or similar
        mask_token="<|mask|>"
    )
    
    print("Tokenizer training complete.")
    return fast_tokenizer

def prepare_pretrain_dataset(
    tokenizer,
    config,
    path: list[str],
):
    """
    Prepare dataset for pretraining.
    If tokenizer is None, it will be trained on the dataset.
    Returns: (dataset, tokenizer)
    """
    
    # Determine max_seq_len from config (handle both dict and object)
    if isinstance(config, dict):
        # Check if 'model_config' exists and has 'max_seq_len'
        if 'model_config' in config and 'max_seq_len' in config['model_config']:
            max_seq_len = config['model_config']['max_seq_len']
        else:
            max_seq_len = config.get("max_seq_len", 2048)
    else:
        max_seq_len = getattr(config, "max_seq_len", 2048)

    print(f"Preparing local dataset for pretraining with max_len={max_seq_len}...")

    # Resolve paths to a list of files
    if isinstance(path, str):
        path = [path]
        
    data_files = []
    for p in path:
        if os.path.isdir(p):
            if os.path.basename(p.rstrip(os.sep)) == 'lm_dataset':
                continue
            # If directory, find all json files recursively
            found = glob(os.path.join(p, "**", "*.json"), recursive=True)
            data_files.extend([f for f in found if "lm_dataset" not in f.replace("\\", "/").split("/")])
        elif os.path.isfile(p):
            if "lm_dataset" not in p.replace("\\", "/").split("/"):
                data_files.append(p)
    
    if not data_files:
        raise ValueError(f"No JSON files found in provided paths: {path}")

    print(f"Found {len(data_files)} JSON files.")

    # 0. Train Tokenizer if needed
    if tokenizer is None:
        vocab_size = config['tokenizer_config'].get("vocab_size", 52000)
        tokenizer = train_tokenizer(data_files, vocab_size=vocab_size)

    print("Loading dataset...")

    # 1. Load Dataset
    def gen():
        for file_path in data_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"⚠️ Error reading {file_path}: {e}")
                continue
            
            data = data['data_info']

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "contents" in item:
                        yield {"text": item["contents"]}
            elif isinstance(data, dict) and "contents" in data:
                yield {"text": data["contents"]}

    dataset = Dataset.from_generator(gen)

    # Filter short data
    dataset = dataset.filter(lambda x: x['text'] is not None and len(x['text']) > 5)

    # 3. Tokenization (Standard Padding)
    print("Tokenizing and Grouping (Chunking) for Pretraining...")
    
    def tokenize_raw(examples):
        return tokenizer(examples["text"], truncation=False, return_attention_mask=False)
    
    tokenized_dataset = dataset.map(
        tokenize_raw,
        batched=True,
        remove_columns=["text"]
    )

    # 4. Grouping
    block_size = max_seq_len
    
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        batch_size=1000,
    )
    
    print(f"DONE. Chunked samples: {len(lm_dataset)}")
    if len(lm_dataset) > 0:
        print(len(lm_dataset[0]['input_ids']))

    return lm_dataset, tokenizer