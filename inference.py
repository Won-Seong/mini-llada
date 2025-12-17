import torch
import os

from safetensors.torch import load_file  # safetensors 사용 시 필요
from transformers import AutoModel, AutoTokenizer

from ko_mini_llada.utils.sampler import Sampler
from ko_mini_llada.models.configuration_mini_llada import MiniLLaDAConfig
from ko_mini_llada.models.modeling_mini_llada import MiniLLaDA

def get_sampler(model_name: str, checkpoint_path=None, device=None):
    print(f"Loading model architecture from Hub: {model_name}")
    
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto" if device is None else {"": device}
    )
    
    if checkpoint_path is not None:
        print(f"Overwriting weights from checkpoint: {checkpoint_path}")
        
        #bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        safe_path = os.path.join(checkpoint_path, "model.safetensors")
        
        state_dict = None
        
        if os.path.exists(safe_path):
            state_dict = load_file(safe_path)
        else:
            raise FileNotFoundError(f"No checkpoint file found in {checkpoint_path}")
            
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    sampler = Sampler(model, tokenizer)
    return sampler