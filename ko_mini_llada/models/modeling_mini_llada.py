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
        self.network = AutoModelForMaskedLM.from_pretrained(config.backbone_model_name)
        self.mask_token_id = config.mask_token_id
        
        # 2. resize token embeddings if needed
        if self.network.config.vocab_size != config.vocab_size:
            self.network.resize_token_embeddings(config.vocab_size)

    def get_input_embeddings(self):
        return self.network.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.network.set_input_embeddings(value)

    def resize_token_embeddings(self, new_num_tokens: int = None, pad_to_multiple_of=None) -> torch.nn.Embedding:
        # Resize the token embeddings of the internal network
        model_embeds = self.network.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # Update the config vocab size
        self.config.vocab_size = model_embeds.num_embeddings
        self.network.config.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # 1. Training Mode
        if labels is not None and self.training:
            # Diffusion Forward Process
            t, noisy_x, mask_indices = self.forward_process(input_ids)
            
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

    def forward_process(self, input_ids):
        B, L = input_ids.shape
        device = input_ids.device

        t = torch.rand(B, device=device)
        mask_probs = t.unsqueeze(1).expand(B, L)
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