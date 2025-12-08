import torch

from ko_mini_llada.utils.sampler import Sampler
from transformers import AutoModel, AutoTokenizer

def get_sampler(model_name: str, checkpoint_path=None, device=None):
    load_path = checkpoint_path if checkpoint_path else model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model = AutoModel.from_pretrained(
        load_path, 
        torch_dtype=torch.float16,
        device_map=device
    )
    
    model.eval()
    sampler = Sampler(model, tokenizer)
    return sampler