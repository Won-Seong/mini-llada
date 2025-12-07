# modeling_llada.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModel, AutoConfig, AutoModelForMaskedLM
from ko_mini_llada.models.configuration_ko_mini_llada import LladaConfig

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
        # Training
        if labels is not None and self.training:
            # 1. Diffusion Forward Process (Adding Noise)
            t, noisy_x, mask_indices = self.forward_process(input_ids)
            
            # 2. Reverse Process
            logits = self.network(noisy_x, attention_mask).logits
            
            # 3. Compute Loss
            loss = self.compute_diffusion_loss(logits, input_ids, mask_indices, attention_mask)
            
            # Trainer는 (loss, logits) 튜플이나 CausalLMOutput 객체를 기대함
            return {"loss": loss, "logits": logits}

        # Inference
        else:
            return self.network(input_ids, attention_mask).logits

    def forward_process(self, input_ids):
        # x is the list of the tokens, shape: [B, L]
        # mask_id is the id of the mask token
        B, L = input_ids.shape
        device = input_ids.device

        # Generate random mask indices based on time steps
        t = torch.rand(B, device=device) # Random time steps for each sample in the batch
        mask_probs = t.unsqueeze(1).expand(B, L) # Shape: [B, L]
        random_matrix = torch.rand(B, L, device=device) # Shape: [B, L]
        mask_indices = (random_matrix < mask_probs) # Shape: [B, L]

        noisy_x = torch.where(mask_indices, self.mask_id, input_ids)
        return t, noisy_x, mask_indices

    def compute_diffusion_loss(self, logits, input_ids, mask_indices, attention_mask):
        B, L, V = logits.shape
        logits_flat = logits.view(-1, V) # Shape: [B*L, V]
        target_flat = input_ids.view(-1) # Shape: [B*L]
        
        target_flat = torch.where(mask_indices.view(-1), target_flat, -100)
        if attention_mask is not None:
            # Apply attention mask to ignore padding tokens in loss calculation
            target_flat = torch.where(attention_mask.view(-1).bool(), target_flat, -100)
        
        return F.cross_entropy(logits_flat, target_flat)