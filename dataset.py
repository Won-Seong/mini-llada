import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

class PreTokenizedDataset(Dataset):
    def __init__(self, input_ids_list):
        self.input_ids = torch.tensor(input_ids_list, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]

def get_tokenizer(model_name="EleutherAI/polyglot-ko-1.3b"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer

def prepare_data(tokenizer, max_seq_len=512):
    dataset = load_dataset("beomi/KoAlpaca-v1.1a", split="train")
    
    # 질문-답변 포맷팅
    texts = [f"질문: {item['instruction']} 답변: {item['output']}" for item in dataset]
    
    
    encodings = tokenizer(
        texts, 
        max_length=max_seq_len, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt" # 나중에 리스트로 변환하지만 처리는 텐서가 빠름
    )
    
    input_ids_list = [ids.tolist() for ids in encodings["input_ids"]]
    
    return PreTokenizedDataset(input_ids_list)