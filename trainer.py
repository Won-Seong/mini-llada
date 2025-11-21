import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from accelerate import Accelerator
import time
import yaml
from tqdm import tqdm

# 우리가 만든 모듈들 import
from network import MiniLLaDA
from dataset import get_tokenizer, prepare_data
from diffusion import DiffusionModel
from helper import init_weights

# ==========================================
# Training Loop
# ==========================================
def main():
    with open("config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)
    
    accelerator = Accelerator(mixed_precision="bf16")
    accelerator.print(f"🚀 Training Start! Device: {accelerator.device}")

    tokenizer = get_tokenizer()
    dataset = prepare_data(tokenizer, max_seq_len=CONFIG['max_seq_len'], dataset_size=CONFIG['dataset_size'])
    
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )

    network = MiniLLaDA(
        vocab_size=len(tokenizer),
        dim=CONFIG["dim"], 
        depth=CONFIG["depth"], 
        heads=CONFIG["heads"],
        intermediate_size=CONFIG["intermediate_size"],
        max_seq_len=CONFIG["max_seq_len"]
    )

    model = DiffusionModel(network)
    
    init_weights(model)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    mask_id = tokenizer.mask_token_id

    model.train()
    
    for epoch in tqdm(range(CONFIG['epochs'])):
        total_loss = 0
        start_time = time.time()
        
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            noisy_batch, mask_indices = model.forward_process(batch, mask_id)
            loss = model.loss(batch, noisy_batch, mask_indices)
            
            accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
            
            if step % 10 == 0:
                avg_step_loss = total_loss / (step + 1)
                accelerator.print(f"\rEpoch {epoch+1} | Step {step}/{len(dataloader)} | Loss: {avg_step_loss:.4f}", end="")
        
        epoch_loss = total_loss / len(dataloader)
        accelerator.print(f"\n✅ Epoch {epoch+1} Complete! Avg Loss: {epoch_loss:.4f} (소요시간: {time.time()-start_time:.1f}초)")
        
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), "mini_llada_model.pth")
        accelerator.print("💾 Model Saved.: mini_llada_model.pth")

if __name__ == "__main__":
    main()