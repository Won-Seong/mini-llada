import torch
from transformers import TrainerCallback
from ko_mini_llada.utils.sampler import Sampler

class GenerateSampleCallback(TrainerCallback):
    def __init__(self, tokenizer, prompt="대한민국의 수도는 어디인가요?", device="cuda"):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.device = device
        
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        # generate sample at the end of each epoch
        print(f"\n[Epoch {state.epoch:.2f}] Generating sample...")
        
        # unwrap model if it is wrapped in DataParallel or DistributedDataParallel
        if hasattr(model, 'module'):
            model = model.module
            
        # Init the sampler and generate
        sampler = Sampler(model, self.tokenizer)
        output = sampler.generate(
            messages=self.prompt,
            steps=16,
            gen_len=16,
            temperature=0.0
        )
        
        print(f"Prompt: {self.prompt}")
        print(f"Generated: {output}")
        print("-" * 50)