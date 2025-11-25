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

def prepare_data(tokenizer, max_seq_len=512, dataset_size=None):
    print(f"⏳ 데이터 로드 및 전처리 중... (Max Len: {max_seq_len})")
    
    all_texts = []

    # 1. 위키백과 (지식 학습용) - 컬럼명: 'text' 등
    # 데이터가 많으니 일부만 로드 (예: 10,000개)
    print("   - 위키백과 로드 중...")
    wiki_data = load_dataset("heegyu/kowiki-paragraphs", split="train") 
    all_texts.extend([item['text'] for item in wiki_data])

    # 2. 교과서 (문법/상식 학습용) - 컬럼명: 'text'
    print("   - 교과서 데이터 로드 중...")
    textbook_data = load_dataset("maywell/korean_textbooks", split="train")
    all_texts.extend([item['text'] for item in textbook_data])

    # 3. KoAlpaca (지시 수행 학습용) - 컬럼명: 'instruction', 'output'
    print("   - KoAlpaca 로드 중...")
    alpaca_data = load_dataset("beomi/KoAlpaca-v1.1a", split="train")
    # Alpaca는 질문-답변 포맷으로 변환해서 추가
    alpaca_texts = [f"질문: {item['instruction']} 답변: {item['output']}" for item in alpaca_data]
    all_texts.extend(alpaca_texts)
    
    print(f"📊 총 데이터 개수: {len(all_texts)}개")
    
    # 4. 토큰화 (한 번에 처리)
    print("⏳ 통합 데이터 토큰화 진행 중...")
    encodings = tokenizer(
        all_texts, 
        max_length=max_seq_len, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )
    
    # 메모리 효율을 위해 리스트로 변환
    input_ids_list = [ids.tolist() for ids in encodings["input_ids"]]
    
    return PreTokenizedDataset(input_ids_list)