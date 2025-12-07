# modeling_llada.py
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel, AutoConfig, AutoModelForMaskedLM
from mini_llada.models.configuration_ko_mini_llada import LladaConfig

class KoMiniLlada(PreTrainedModel):
    config_class = LladaConfig

    def __init__(self, config: LladaConfig):
        super().__init__(config)
        self.config = config
        
        # 1. 
        backbone_config = AutoConfig.from_pretrained(config.backbone_model_name)
        # 
        self.backbone = AutoModel.from_config(backbone_config)
        
        # 2. 
        self.network = AutoModelForMaskedLM.from_pretrained(config.backbone_model_name)
        self.mask_token_id = config.mask_token_id
        
        # 
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # [학습 모드] Trainer가 labels를 넘겨줬을 때
        if labels is not None and self.training:
            # 1. Diffusion Forward Process (노이즈 주입)
            # 기존 diffusion.py의 forward_process 로직을 여기로 가져옴
            t, noisy_x, mask_indices = self.forward_process(input_ids)
            
            # 2. Reverse Process (Logit 예측)
            logits = self.network(noisy_x, attention_mask).logits
            
            # 3. Loss 계산
            # 기존 diffusion.py의 loss 계산 로직
            loss = self.compute_diffusion_loss(logits, input_ids, mask_indices, attention_mask)
            
            # Trainer는 (loss, logits) 튜플이나 CausalLMOutput 객체를 기대함
            return {"loss": loss, "logits": logits}

        # [추론 모드] 그냥 Logit만 반환
        else:
            return self.network(input_ids, attention_mask).logits