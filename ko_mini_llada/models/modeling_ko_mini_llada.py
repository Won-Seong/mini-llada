# modeling_llada.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
from ko_mini_llada.models.configuration_ko_mini_llada import LladaConfig

class KoMiniLlada(PreTrainedModel):
    config_class = LladaConfig

    def __init__(self, config: LladaConfig):
        super().__init__(config)
        self.config = config
        
        # [수정] backbone과 network 중복 제거 -> network만 유지
        # AutoModelForMaskedLM은 Encoder + LM Head를 모두 포함하므로 이것만 있으면 됩니다.
        self.network = AutoModelForMaskedLM.from_pretrained(config.backbone_model_name)
        
        self.mask_token_id = config.mask_token_id
        
        # 가중치 초기화 (PreTrainedModel 기능)
        self.post_init()

    # [추가해야 할 메서드 1] 임베딩 층 가져오기 
    def get_input_embeddings(self):
        # self.network는 AutoModelForMaskedLM이므로, 그 안의 임베딩을 찾아 반환
        return self.network.get_input_embeddings()

    # [추가해야 할 메서드 2] 임베딩 층 설정하기 (Resize 시 필요)
    def set_input_embeddings(self, value):
        self.network.set_input_embeddings(value)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # 1. Training Mode
        if labels is not None and self.training:
            # Diffusion Forward Process
            t, noisy_x, mask_indices = self.forward_process(input_ids)
            
            # Reverse Process (Logit 예측)
            # network outputs: MaskedLMOutput (logits, hidden_states, etc.)
            outputs = self.network(input_ids=noisy_x, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Compute Loss
            loss = self.compute_diffusion_loss(logits, input_ids, mask_indices, attention_mask)
            
            # [추천] HF Trainer 호환성을 위해 Dictionary 대신 Output 객체나 튜플 반환
            return MaskedLMOutput(loss=loss, logits=logits)

        # 2. Inference Mode
        else:
            outputs = self.network(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits

    def forward_process(self, input_ids):
        B, L = input_ids.shape
        device = input_ids.device

        t = torch.rand(B, device=device)
        mask_probs = t.unsqueeze(1).expand(B, L)
        random_matrix = torch.rand(B, L, device=device)
        mask_indices = (random_matrix < mask_probs)

        noisy_x = torch.where(mask_indices, self.mask_token_id, input_ids) # self.mask_token_id 사용
        return t, noisy_x, mask_indices

    def compute_diffusion_loss(self, logits, input_ids, mask_indices, attention_mask):
        B, L, V = logits.shape
        logits_flat = logits.view(-1, V)
        target_flat = input_ids.view(-1)
        
        # 마스킹된 부분만 Loss 계산
        target_flat = torch.where(mask_indices.view(-1), target_flat, -100)
        
        if attention_mask is not None:
            target_flat = torch.where(attention_mask.view(-1).bool(), target_flat, -100)
        
        return F.cross_entropy(logits_flat, target_flat)