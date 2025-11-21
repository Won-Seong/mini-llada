import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, x):
        return self.reverse_process(x)

    def forward_process(self, x, mask_id):
        # x is the list of the tokens, shape: [B, L]
        # mask_id is the id of the mask token
        B, L = x.shape
        device = x.device

        # Generate random mask indices based on time steps
        t = torch.rand(B, device=device) # Random time steps for each sample in the batch
        mask_probs = t.unsqueeze(1).expand(B, L) # Shape: [B, L]
        random_matrix = torch.rand(B, L, device=device) # Shape: [B, L]
        mask_indices = (random_matrix < mask_probs) # Shape: [B, L]

        noisy_x = torch.where(mask_indices, mask_id, x)
        return noisy_x, mask_indices

    def reverse_process(self, noisy_x):
        logits = self.network(noisy_x)
        return logits

    def loss(self, x, noisy_x, mask_indices):
        logits = self.reverse_process(noisy_x)

        B, L, V = logits.shape
        
        logits_flat = logits.view(-1, V)
        target_flat = x.view(-1)
        
        target_flat = torch.where(mask_indices.view(-1), target_flat, -100)
        
        loss = F.cross_entropy(logits_flat, target_flat) 
        return loss