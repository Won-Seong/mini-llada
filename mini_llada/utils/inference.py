import torch
import os

from mini_llada.models.network import get_pretrained_model, Wrapper
from mini_llada.data.dataset import get_tokenizer
from mini_llada.models.diffusion import DiffusionModel
from mini_llada.utils.sampler import Sampler

class Inferencer:
    def __init__(self, config:dict, checkpoint_path=None, device=None):
        self.config = config

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"Inference initialized on {self.device}")

        self.tokenizer = get_tokenizer(self.config['pretrained_model_name'])
        if self.tokenizer.mask_token_id is None:
            self.tokenizer.add_special_tokens({'mask_token': '[MASK]'})
            self.tokenizer.mask_token_id = self.tokenizer.convert_tokens_to_ids('[MASK]')

        self.network = Wrapper(get_pretrained_model(self.config['pretrained_model_name']))
        self.model = DiffusionModel(self.network, self.tokenizer.mask_token_id)

        self.model.to(self.device)
        self.model.eval()

        if checkpoint_path is not None:
            self.load_model(checkpoint_path)

        self.sampler = Sampler(self.model, self.tokenizer)

    def load_model(self, checkpoint_path: str):
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Model loaded from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    def generate_text(self, prompt: str, steps: int = 32, gen_len: int = 32, 
                      temperature: float = 0.8, repetition_penalty: float = 1.0, 
                      print_progress: bool = False) -> str:
        generated_text = self.sampler.generate(
            prompt_text=prompt,
            steps=steps,
            gen_len=gen_len,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            print_progress=print_progress
        )
        return generated_text