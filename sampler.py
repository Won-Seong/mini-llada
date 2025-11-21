import torch
from diffusion import DiffusionModel

class Sampler():
    def __init__(self, model: DiffusionModel, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_id = tokenizer.mask_token_id
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def generate(self, prompt_text, steps: int = 32, gen_len: int = 32, print_progress: bool = False):
        self.model.eval()

        prompts_ids = self.tokenizer.encode(prompt_text, return_tensors='pt').to(self.device) # Shape: [1, L]
        mask_tokens = torch.full((1, gen_len), self.mask_id, dtype=torch.long, device=self.device) # Shape: [1, gen_len]
        x = torch.cat([prompts_ids, mask_tokens], dim=1)
        L = gen_len # Length of the generated part
        prompt_len = prompts_ids.size(1)

        timesteps = torch.linspace(1, 0, steps + 1)[:-1] # Shape: [steps]

        for i, t in enumerate(timesteps):
            s = t - (1.0 / steps) # Previous timestep
            if s < 0:
                s = 0.0
            
            # Unmasking
            logits = self.model(x)
            gen_logits = logits[:, prompt_len:, :] # Shape: [1, L, vocab_size]
            probs = torch.softmax(gen_logits / 1.0, dim=-1) # Temperature = 1.0
            pred_ids = torch.argmax(probs, dim=-1) # Shape: [1, L]
            confidence = torch.max(probs, dim=-1).values # Shape: [1, L]

            # Remasking
            n_keep = int(L * (1 - s))
            n_keep = max(n_keep, int(L * (1 - t)) + 1)
            n_keep = min(n_keep, L) # Max n_keep is L

            x_next = x.clone()
            x_next[:, prompt_len:] = self.pred_ids

            if n_keep < L:
                _, keep_indices = torch.topk(confidence, k=n_keep, dim=-1)
                new_mask_section = torch.full((1, L), self.mask_id, device=self.device)
                new_mask_section.scatter_(1, keep_indices, pred_ids.gather(1, keep_indices))
                x[:, prompt_len:] = new_mask_section
            else:
                # n_keep == L, no masking needed
                x = x_next

            if print_progress:
                curr_text = self.tokenizer.decode(x[0], skip_special_tokens=False)
                print(f"Step {i+1}/{steps} (t={t:.2f} -> s={s:.2f}): ...{curr_text[-30:]}")
        
        return self.tokenizer.decode(x[0], skip_special_tokens=True)
            