import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset, concat_datasets

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

def prepare_data(tokenizer, dataset_config: list[dict], max_seq_len: int = 512):
    print(f"Preparing dataset with max_len={max_seq_len}...")
    processed_datasets = []

    for config in dataset_config:
        name = config['name']
        split = config.get('split', 'train')
        limit = config.get('limit', None)
        q_col = config['question_column']
        a_col = config['answer_column']

        print(f"Loading dataset: {name}...")

        try:
            # 1. load dataset
            dataset = load_dataset(name, split=split)
            if limit:
                dataset = dataset.select(range(limit))

            # 2. format dataset
            def format_example(example):
                return {'text': f"질문: {example[q_col]} 답변: {example[a_col]}"}

            dataset = dataset.map(format_example, remove_columns=dataset.column_names)
            processed_datasets.append(dataset)
        except Exception as e:
            print(f"Error loading dataset {name}: {e}")
            continue

    if not processed_datasets:
        raise ValueError("No datasets were successfully loaded.")
    
    # 3. Combine all datasets
    combined_dataset = concat_datasets(processed_datasets)

    # 4. Tokenization
    print("Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_seq_len,
            return_tensors="pt"
        )

    tokenized_dataset = combined_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format("torch")
    print("Dataset preparation complete. Number of samples:", len(tokenized_dataset))
    return tokenized_dataset