# modeling_llada.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
from ko_mini_llada.models.configuration_mini_llada import MiniLLaDAConfig

class MiniLLaDA(PreTrainedModel):
    config_class = MiniLLaDAConfig

    def __init__(self, config: MiniLLaDAConfig):
        super().__init__(config)
        
        # 1. load backbone model
        self.network = AutoModelForMaskedLM.from_pretrained(config.backbone_model_name, trust_remote_code=True)
        self.mask_token_id = config.mask_token_id

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # 1. Training and Evaluation Mode
        if labels is not None:
            # Diffusion Forward Process
            t, noisy_x, mask_indices = self.forward_process(input_ids, labels)
            
            # Reverse Process
            # network outputs: MaskedLMOutput (logits, hidden_states, etc.)
            outputs = self.network(input_ids=noisy_x, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Compute Loss
            loss = self.compute_diffusion_loss(logits, input_ids, mask_indices, attention_mask)
            
            return MaskedLMOutput(loss=loss, logits=logits)

        # 2. Inference Mode
        else:
            outputs = self.network(input_ids=input_ids, attention_mask=attention_mask)
            return MaskedLMOutput(logits=outputs.logits)

    def forward_process(self, input_ids, labels=None):
        B, L = input_ids.shape
        device = input_ids.device

        t = torch.rand(B, device=device)
        mask_probs = t.unsqueeze(1).expand(B, L)

        if labels is not None:
            train_mask = (labels != -100).float() 
            mask_probs = mask_probs * train_mask  # Make the probabilities zero where labels == -100, which is user message parts.

        random_matrix = torch.rand(B, L, device=device)
        mask_indices = (random_matrix < mask_probs)

        noisy_x = torch.where(mask_indices, self.mask_token_id, input_ids)
        return t, noisy_x, mask_indices

    def compute_diffusion_loss(self, logits, input_ids, mask_indices, attention_mask):
        B, L, V = logits.shape
        logits_flat = logits.view(-1, V)
        target_flat = input_ids.view(-1)
        
        # Calculate loss only for masked positions
        target_flat = torch.where(mask_indices.view(-1), target_flat, -100)
        
        if attention_mask is not None:
            target_flat = torch.where(attention_mask.view(-1).bool(), target_flat, -100)
        
        return F.cross_entropy(logits_flat, target_flat)