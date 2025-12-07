import torch
import os

from ko_mini_llada.models.configuration_ko_mini_llada import LladaConfig
from ko_mini_llada.models.modeling_ko_mini_llada import KoMiniLlada
from ko_mini_llada.data.dataset import get_tokenizer
from ko_mini_llada.utils.sampler import Sampler

class Inferencer:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.config = LladaConfig()
        self.model = KoMiniLlada(self.config)

        print(f"Inference initialized on {self.device}")

        self.tokenizer = get_tokenizer(self.config['pretrained_model_name'])
        self.sampler = Sampler(self.model, self.tokenizer)

    def generate_text(self, prompt: str, steps: int = 32, gen_len: int = 32, 
                      temperature: float = 0.8, print_progress: bool = False) -> str:
        generated_text = self.sampler.generate(
            prompt_text=prompt,
            steps=steps,
            gen_len=gen_len,
            temperature=temperature,
            print_progress=print_progress
        )
        return generated_text