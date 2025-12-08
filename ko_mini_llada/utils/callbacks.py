import torch
from transformers import TrainerCallback
from ko_mini_llada.utils.sampler import Sampler # 사용자님의 Sampler 재사용

class GenerateSampleCallback(TrainerCallback):
    def __init__(self, tokenizer, prompt="대한민국의 수도는 어디인가요?", device="cuda"):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.device = device
        
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        # 평가(Evaluation)가 끝날 때마다 실행됨
        print(f"\n[Epoch {state.epoch:.2f}] Generating sample...")
        
        # 모델이 DDP 등으로 감싸져 있을 수 있으므로 unwrap
        if hasattr(model, 'module'):
            model = model.module
            
        # Sampler 초기화 및 생성
        sampler = Sampler(model, self.tokenizer)
        output = sampler.generate(
            prompt_text=self.prompt,
            steps=32,
            gen_len=64,
            temperature=0.0 # 정성 평가를 위해 Greedy 사용
        )
        
        print(f"Prompt: {self.prompt}")
        print(f"Generated: {output}")
        print("-" * 50)