import torch
import torch.nn as nn

class DiffusionModel(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward_process(self, x, mask_id):
        # x is the list of the tokens, shape: [B, L]
        # mask_id is the id of the mask token
        B, L = x.shape
        device = x.device

        # Generate random mask indices based on time steps
        t = torch.rand(B, device=device) # Random time steps for each sample in the batch
        mask_probs = t.unsqueeze(1).expand(B, L) # Shape: [B, L]
        random_matrix = torch.rand(B, L, device=device) # Shape: [B, L]
        mask_indices = (random_matrix < mask_probs).long() # Shape: [B, L]

        noisy_x = torch.where(mask_indices, mask_id, x)
        return noisy_x, mask_indices

    def reverse_process(self, noisy_x, mask_indices, mask_id):
        # noisy_x is the input with masked tokens, shape: [B, L]
        # mask_indices indicates which tokens were masked, shape: [B, L]
        # mask_id is the id of the mask token
        B, L = noisy_x.shape
        device = noisy_x.device

        # Predict the original tokens using the network
        logits = self.network(noisy_x) # Shape: [B, L, vocab_size]
        predicted_tokens = torch.argmax(logits, dim=-1) # Shape: [B, L]

        # Replace only the masked positions with predicted tokens
        reconstructed_x = torch.where(mask_indices, predicted_tokens, noisy_x)
        return reconstructed_x