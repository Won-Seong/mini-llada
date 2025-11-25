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
    print("⏳ 데이터셋 로드 및 처리 중...")
    
    # 1. 데이터셋들을 로드하고 합칩니다.
    wiki_data = load_dataset("maywell/ko_wikidata_QA", split="train[:10000]")
    gpt_data = load_dataset("maywell/ko-gpt3_14k", split="train[:10000]")
    alpaca_data = load_dataset("beomi/KoAlpaca-v1.1a", split="train[:10000]")
    
    # 2. 포맷팅 함수 (제너레이터로 처리해서 메모리 아낌)
    def format_wiki(example): return {'text': f"질문: {example['instruction']} 답변: {example['output']}"}
    def format_gpt(example): return {'text': f"질문: {example['question']} 답변: {example['answer']}"}
    def format_alpaca(example): return {'text': f"질문: {example['instruction']} 답변: {example['output']}"}
    
    wiki_data = wiki_data.map(format_wiki, remove_columns=wiki_data.column_names)
    gpt_data = gpt_data.map(format_gpt, remove_columns=gpt_data.column_names)
    alpaca_data = alpaca_data.map(format_alpaca, remove_columns=alpaca_data.column_names)
    
    # 합치기
    from datasets import concatenate_datasets
    dataset = concatenate_datasets([wiki_data, gpt_data, alpaca_data])
    
    # 3. 토크나이징 (배치 단위로 처리해서 메모리 터짐 방지)
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_seq_len,
            return_tensors="pt"
        )
    
    # batched=True로 하면 알아서 나눠서 처리함
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # PyTorch 텐서로 포맷 설정
    tokenized_dataset.set_format("torch")
    
    return tokenized_dataset

# def prepare_data(tokenizer, max_seq_len=512, dataset_size=None):
#     print(f"⏳ 데이터 로드 및 전처리 중... (Max Len: {max_seq_len})")
    
#     all_texts = []

#     # 1. 위키백과 (지식 학습용) - 컬럼명: 'text' 등
#     # 데이터가 많으니 일부만 로드 (예: 10,000개)
#     print("   - 위키백과 로드 중...")
#     wiki_data = load_dataset("maywell/ko_wikidata_QA", split="train")
#     wiki_texts = [f"질문: {item['instruction']} 답변: {item['output']}" for item in wiki_data] 
#     all_texts.extend(wiki_texts)

#     # 2. Ko - 컬럼명: 'text'
#     print("   - 교과서 데이터 로드 중...")
#     gpt_data = load_dataset("maywell/ko-gpt3_14k", split="train")
#     gpt_texts = [f"질문: {item['question']} 답변: {item['answer']}" for item in gpt_data]
#     all_texts.extend(gpt_texts)

#     # 3. KoAlpaca (지시 수행 학습용) - 컬럼명: 'instruction', 'output'
#     print("   - KoAlpaca 로드 중...")
#     alpaca_data = load_dataset("beomi/KoAlpaca-v1.1a", split="train")
#     # Alpaca는 질문-답변 포맷으로 변환해서 추가
#     alpaca_texts = [f"질문: {item['instruction']} 답변: {item['output']}" for item in alpaca_data]
#     all_texts.extend(alpaca_texts)
    
#     print(f"📊 총 데이터 개수: {len(all_texts)}개")
    
#     # 4. 토큰화 (한 번에 처리)
#     print("⏳ 통합 데이터 토큰화 진행 중...")
#     encodings = tokenizer(
#         all_texts, 
#         max_length=max_seq_len, 
#         padding="max_length", 
#         truncation=True, 
#         return_tensors="pt"
#     )
    
#     # 메모리 효율을 위해 리스트로 변환
#     input_ids_list = [ids.tolist() for ids in encodings["input_ids"]]
    
#     return PreTokenizedDataset(input_ids_list)