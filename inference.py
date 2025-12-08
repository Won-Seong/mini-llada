import torch
import os

from ko_mini_llada.utils.sampler import Sampler
from transformers import AutoModel, AutoTokenizer

def get_sampler(model_name:str, checkpoint_path=None, device=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    sampler = Sampler(model, tokenizer)
    return sampler