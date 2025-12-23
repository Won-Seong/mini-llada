import os
import json
from glob import glob
from itertools import chain
from datasets import load_dataset, Dataset

def prepare_dataset(
    tokenizer,
    config,
    path: list[str],
):
    # Determine max_seq_len from config (handle both dict and object)
    if isinstance(config, dict):
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
            # If directory, find all json files
            found = glob(os.path.join(p, "*.json"))
            data_files.extend(found)
        elif os.path.isfile(p):
            data_files.append(p)
        else:
            # Handle glob patterns if passed as string
            found = glob(p)
            data_files.extend(found)
    
    if not data_files:
        raise ValueError(f"No JSON files found in provided paths: {path}")

    print(f"Found {len(data_files)} JSON files. Loading...")

    # 1. Load Dataset
    # Use a generator to handle potential schema inconsistencies or JSON errors
    def gen():
        for file_path in data_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
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
    return lm_dataset