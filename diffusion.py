import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    def __init__(self, network: nn.Module, mask_id: int):
        super().__init__()
        self.network = network
        self.mask_id = mask_id

    def forward(self, x, attention_mask=None):
        return self.reverse_process(x, attention_mask)

    def forward_process(self, x):
        # x is the list of the tokens, shape: [B, L]
        # mask_id is the id of the mask token
        B, L = x.shape
        device = x.device

        # Generate random mask indices based on time steps
        t = torch.rand(B, device=device) # Random time steps for each sample in the batch
        mask_probs = t.unsqueeze(1).expand(B, L) # Shape: [B, L]
        random_matrix = torch.rand(B, L, device=device) # Shape: [B, L]
        mask_indices = (random_matrix < mask_probs) # Shape: [B, L]

        noisy_x = torch.where(mask_indices, self.mask_id, x)
        return t, noisy_x, mask_indices

    def reverse_process(self, noisy_x, attention_mask=None):
        logits = self.network(noisy_x,  attention_mask=attention_mask)
        return logits

    def loss(self, x, t, noisy_x, mask_indices, attention_mask):
        logits = self(noisy_x, attention_mask) # Shape: [B, L, V]

        B, L, V = logits.shape
        logits_flat = logits.view(-1, V) # Shape: [B*L, V]
        target_flat = x.view(-1) # Shape: [B*L]
        
        target_flat = torch.where(mask_indices.view(-1), target_flat, -100)
        if attention_mask is not None:
            # Apply attention mask to ignore padding tokens in loss calculation
            target_flat = torch.where(attention_mask.view(-1).bool(), target_flat, -100)
        
        return F.cross_entropy(logits_flat, target_flat)