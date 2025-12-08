import torch

from ko_mini_llada.utils.sampler import Sampler
from transformers import AutoModel, AutoTokenizer, AutoConfig

def get_sampler(model_name: str, checkpoint_path=None, device=None):
    load_path = checkpoint_path if checkpoint_path else model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    model = AutoModel.from_pretrained(
        load_path,              # 가중치 위치 (체크포인트 폴더)
        config=config,          # 구조 정보 (원본 모델 설정)
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True  # 원본 config에 연결된 원격 코드를 허용
    )
    
    model.eval()
    sampler = Sampler(model, tokenizer)
    return sampler