import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

class PreTokenizedDataset(Dataset):
    def __init__(self, input_ids_list):
        self.input_ids = torch.tensor(input_ids_list, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]

def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)    
    return tokenizer

def prepare_dataset(
    tokenizer, 
    dataset_config: list[dict], 
    max_seq_len: int = 512, 
    mode: str = "pretrain",
):
    """
    Args:
        mode (str): 'pretrain' or 'sft'
    """
    print(f"Preparing dataset for [{mode}] with max_len={max_seq_len}...")
    processed_datasets = []

    for config in dataset_config:
        name = config['name']
        subset = config.get('subset', None)
        split = config.get('split', 'train')
        limit = config.get('limit', None)
        
        print(f"Loading dataset: {name} (subset: {subset})...")

        try:
            # 1. Load Dataset
            if subset:
                dataset = load_dataset(name, subset, split=split)
            else:
                dataset = load_dataset(name, split=split)

            if limit:
                dataset = dataset.select(range(limit))

            # 2. Format Text
            if mode == "sft":
                # Fine-tuning
                q_col = config.get('q_col', 'question')
                a_col = config.get('a_col', 'answer')
                
                def format_sft(example):
                    return {'text': f"질문: {example[q_col]}\n답변: {example[a_col]}"}
                
                dataset = dataset.map(format_sft, remove_columns=dataset.column_names)
                
            else: # mode == "pretrain"
                # Pre-training
                text_col = config.get('text_col', 'text')
                
                def format_pretrain(example):
                    return {'text': example[text_col]}
                
                dataset = dataset.map(format_pretrain, remove_columns=dataset.column_names)

            # remove short or empty texts
            dataset = dataset.filter(lambda x: x['text'] is not None and len(x['text']) > 5)
            processed_datasets.append(dataset)

        except Exception as e:
            print(f"⚠️ Error loading dataset {name}: {e}")
            continue

    if not processed_datasets:
        raise ValueError("No datasets were successfully loaded.")
    
    # 3. Combine
    combined_dataset = concatenate_datasets(processed_datasets)

    # 4. Tokenization
    print("Tokenizing dataset...")
    def tokenize_function(examples):
        if mode == 'pretrain':
            return tokenizer(
                examples["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=max_seq_len,
                return_tensors="pt"
            )
        else: # mode == 'sft'
            # 1. without padding first
            tokenized = tokenizer(
                examples["text"], 
                truncation=True, 
                max_length=max_seq_len, 
                padding=False, 
                return_attention_mask=False
            )
            
            new_input_ids = []
            new_attention_masks = []
            
            for ids in tokenized["input_ids"]:
                curr_len = len(ids)
                pad_len = max_seq_len - curr_len # padding length
                
                # 2. add eos token for padding
                final_ids = ids + [tokenizer.eos_token_id] * pad_len
                final_mask = [1] * max_seq_len # attention mask
                
                new_input_ids.append(final_ids)
                new_attention_masks.append(final_mask)
            
            return {
                "input_ids": new_input_ids, 
                "attention_mask": new_attention_masks
            }

    tokenized_dataset = combined_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    print(f"DONE. Total samples: {len(tokenized_dataset)}")
    return tokenized_dataset