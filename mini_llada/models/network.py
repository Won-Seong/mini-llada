import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, attention_mask=None):
        outputs = self.model(input_ids=x, attention_mask=attention_mask, output_hidden_states=False)
        return outputs.logits

def get_pretrained_model(pretrained_model_name:str):
    print(f"⏳ Loading {pretrained_model_name} with 4-bit Quantization...")
    
    # 1. 4비트 양자화 설정 (메모리 절약)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # 2. 모델 로드 (CausalLM으로 불러오지만 마스크를 조작해서 씀)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name,
        quantization_config=bnb_config,
        device_map="auto" # 알아서 GPU에 분배
    )
    
    # 3. LoRA 설정 (훈련 가능한 작은 파라미터 붙이기)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=16,            # Rank (클수록 파라미터 많아짐)
        lora_alpha=32, 
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
    )
    
    # 4. 모델에 LoRA 장착
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters() # 훈련할 파라미터 수 출력 (약 0.1% ~ 1%)
    
    return model