import torch
from mini_llada.models.diffusion import DiffusionModel

class Sampler():
    def __init__(self, model: DiffusionModel, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_id = tokenizer.mask_token_id
        self.device = next(model.parameters()).device

    def get_num_transfer_tokens(self, mask_index, steps):
        """
        공식 코드의 스케줄링 로직 포팅:
        남은 마스크 개수를 스텝 수로 나누어, 이번 스텝에 벗겨야 할 정확한 개수를 계산
        """
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        
        # 기본적으로 base만큼 벗김
        num_transfer = torch.zeros_like(mask_num) + base
        
        # 나머지가 있으면 앞쪽 배치부터 1개씩 더 벗김 (배치 단위 처리 시 필요)
        # 현재 배치가 1개라면 remainder가 있으면 +1, 없으면 +0
        if remainder > 0:
            num_transfer += 1
            
        return num_transfer

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
        # LLaDA는 남은 스텝 수를 기준으로 계산하므로, range(steps)로 순회
        for i in range(steps):
            # 현재 스텝에서 처리해야 할 남은 스텝 수 (ex: 32 -> 31 -> ... -> 1)
            steps_left = steps - i
            
            # 현재 마스크 위치 확인 (Prompt 부분은 제외하고 생성 부분만)
            # x 전체에서 마스크인 부분 찾기
            mask_index = (x == self.mask_id)
            
            # 생성 부분(gen_len)에 남아있는 마스크 개수 확인
            gen_mask_index = mask_index[:, prompt_len:]
            
            # 이번 스텝에 Unmasking 할 개수 계산 (Linear Schedule)
            num_transfer = self.get_num_transfer_tokens(gen_mask_index, steps_left)
            
            # 3. Model Prediction
            logits = self.model(x).logits

            # 4. Sampling (Gumbel Noise or Temperature)
            # 공식 코드는 float64 변환 후 Gumbel Noise 사용을 권장
            if temperature > 0:
                logits = logits.to(torch.float64)
                noise = torch.rand_like(logits)
                gumbel_noise = (-torch.log(noise + 1e-20)) # numerical stability
                logits_with_noise = (torch.log(torch.softmax(logits, dim=-1) + 1e-20) + gumbel_noise) / temperature
                x0 = torch.argmax(logits_with_noise, dim=-1)
            else:
                x0 = torch.argmax(logits, dim=-1)

            # 5. Confidence Calculation (Low Confidence Strategy)
            # 현재 예측된 토큰의 확률(Confidence) 계산
            probs = torch.softmax(logits, dim=-1)
            # x0(예측된 토큰)의 확률만 가져옴
            x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

            # 6. Update (Monotonic Unmasking)
            # 중요: 이미 마스크가 벗겨진 곳(prompt 포함)은 건드리지 않음 (-inf로 설정하여 선택 안되게 함)
            # mask_index가 False인 곳(이미 단어가 있는 곳)의 confidence를 -inf로
            confidence = torch.where(mask_index, x0_p, torch.tensor(float('-inf')).to(self.device))
            
            # 생성 영역(prompt 이후)에서만 Top-k 선택
            # 전체 시퀀스 길이에서 선택하지만, prompt 부분은 이미 -inf라서 선택 안됨
            # 이번 스텝에 벗길 개수(num_transfer)만큼 가장 자신 있는 마스크 선택
            _, select_index = torch.topk(confidence, k=num_transfer.item())
            
            # 선택된 인덱스만 x0(예측값)로 업데이트
            # 마스킹 되어있던 곳 -> 예측값으로 변환 (확정)
            # scatter나 인덱싱 활용. 여기선 간단히 loop나 boolean indexing 사용 가능하지만
            # 배치 1개 가정하에 간단히:
            x[0, select_index] = x0[0, select_index]

            if print_progress:
                curr_text = self.tokenizer.decode(x[0, prompt_len:], skip_special_tokens=False)
                print(f"Step {i+1}/{steps}: {curr_text}")
        
        return self.tokenizer.decode(x[0, prompt_len:], skip_special_tokens=True)
            