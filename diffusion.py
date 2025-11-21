import torch
import torch.nn as nn

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

        # Compute loss only on masked positions
        loss_fn = nn.CrossEntropyLoss()
        masked_logits = logits[mask_indices.bool()] # Shape: [N_masked, vocab_size]
        masked_targets = x[mask_indices.bool()]    # Shape: [N_masked]

        loss = loss_fn(masked_logits, masked_targets)
        return loss