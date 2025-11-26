import torch
from diffusion import DiffusionModel

class Sampler():
    def __init__(self, model: DiffusionModel, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_id = tokenizer.mask_token_id
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def generate(self, prompt_text, steps: int = 32, gen_len: int = 32, temperature=0.8,
                 print_progress: bool = False):
        """
        Generates text using the diffusion model given a prompt.
        """
        self.model.eval()

        prompts_ids = self.tokenizer.encode(prompt_text, return_tensors='pt').to(self.device) # Shape: [1, prompt_len]
        mask_tokens = torch.full((1, gen_len), self.mask_id, dtype=torch.long, device=self.device) # Shape: [1, gen_len]
        x = torch.cat([prompts_ids, mask_tokens], dim=1)
        L = x.size(1) # Total length (prompt + gen_len)
        prompt_len = prompts_ids.size(1) # Length of the prompt
        gen_len = mask_tokens.size(1) # Length of the generation part

        timesteps = torch.linspace(1, 0, steps + 1)[:-1] # Shape: [steps]

        for i, t in enumerate(timesteps):
            s = t - (1.0 / steps) # Previous timestep
            if s < 0:
                s = 0.0
            
            # Unmasking
            logits = self.model(x)
            gen_logits = logits[:, prompt_len:, :] # Shape: [1, gen_len, vocab_size]
            if temperature > 0:
                probs = torch.softmax(gen_logits / temperature, dim=-1)
                pred_ids = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(1, gen_len) 
                confidence = torch.gather(probs, -1, pred_ids.unsqueeze(-1)).squeeze(-1)
            else:
                probs = torch.softmax(gen_logits, dim=-1)
                pred_ids = torch.argmax(probs, dim=-1)
                confidence = torch.max(probs, dim=-1).values

            # Remasking
            n_remask = min(int(gen_len * (s / t)), gen_len)
            n_keep = int(gen_len - n_remask)

            x_next = x.clone()
            x_next[:, prompt_len:] = pred_ids

            if n_keep < gen_len:
                _, keep_indices = torch.topk(confidence, k=n_keep, dim=-1)
                new_mask_section = torch.full((1, gen_len), self.mask_id, device=self.device)
                new_mask_section.scatter_(1, keep_indices, pred_ids.gather(1, keep_indices))
                x[:, prompt_len:] = new_mask_section
            else:
                # n_keep == gen_len, no masking needed
                x = x_next

            if print_progress:
                curr_text = self.tokenizer.decode(x[0], skip_special_tokens=False)
                print(f"Step {i+1}/{steps} (t={t:.2f} -> s={s:.2f}): ...{curr_text[-30:]}")
        
        return self.tokenizer.decode(x[0], skip_special_tokens=True)
            