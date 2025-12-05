import torch
from mini_llada.models.diffusion import DiffusionModel

class Sampler():
    def __init__(self, model: DiffusionModel, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_id = tokenizer.mask_token_id
        self.device = next(model.parameters()).device

    def add_gumbel_noise(self, logits, temperature):
        '''
        https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
        '''
        if temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (- torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def get_num_transfer_tokens(self, mask_index, steps):
        '''
        https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
        '''
        mask_num = mask_index.sum(dim=1, keepdim=True)

        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, :remainder[i]] += 1

        return num_transfer_tokens

    @torch.no_grad()
    def generate(self, prompt_text, steps: int = 32, gen_len: int = 128, 
                 temperature=0.0, print_progress: bool = False):
        
        self.model.eval()

        # 1. Input Setup
        prompts_ids = self.tokenizer.encode(prompt_text, return_tensors='pt').to(self.device)
        prompt_len = prompts_ids.size(1)
        
        mask_tokens = torch.full((1, gen_len), self.mask_id, dtype=torch.long, device=self.device)
        x = torch.cat([prompts_ids, mask_tokens], dim=1) # Shape: [1, prompt_len + gen_len]

        # 2. Step Loop
        for i in range(steps):
            steps_left = steps - i
            
            # mask index
            mask_index = (x == self.mask_id)
            
            # mask index for generation part only
            gen_mask_index = mask_index[:, prompt_len:]
            
            # number of tokens to transfer this step
            num_transfer = self.get_num_transfer_tokens(gen_mask_index, steps_left)
            
            # 3. Model Prediction
            logits = self.model(x)  # Shape: [1, seq_len, vocab_size]

            # 4. Sampling (Gumbel Noise or Temperature)
            logits = logits.to(torch.float64)
            logits_with_noise = self.add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # 5. Confidence Calculation (Low Confidence Strategy)
            # calculate probabilities
            probs = torch.softmax(logits, dim=-1)
            # get probabilities of predicted tokens
            x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            # 6. Update (Monotonic Unmasking)
            confidence = torch.where(mask_index, x0_p, torch.tensor(float('-inf')).to(self.device))
            
            # Select indices with lowest confidence
            _, select_index = torch.topk(confidence, k=num_transfer.item())
            
            # Update x at selected indices
            x[0, select_index] = x0[0, select_index]

            if print_progress:
                curr_text = self.tokenizer.decode(x[0, prompt_len:], skip_special_tokens=False)
                print(f"Step {i+1}/{steps}: {curr_text}")
        
        return self.tokenizer.decode(x[0, prompt_len:], skip_special_tokens=True)
            