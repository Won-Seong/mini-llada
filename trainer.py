import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from accelerate import Accelerator
import time
from tqdm import tqdm
import os

from mini_llada.models.network import get_pretrained_bert_model, BERT_Wrapper
from mini_llada.data.dataset import get_tokenizer, prepare_dataset
from mini_llada.models.diffusion import DiffusionModel

class Trainer:
    def __init__(self, config:dict):
        self.config = config
        self.global_step = 0
        self.start_epoch = 1
        self.best_valid_loss = float('inf')
        
        self.accelerator = Accelerator(
            mixed_precision="bf16",
            gradient_accumulation_steps=self.config['train_config'].get('gradient_accumulation_steps', 1))
        self.accelerator.print(f"Training Start! Device: {self.accelerator.device}")

        self.tokenizer = get_tokenizer(self.config['pretrained_model_name'])
        dataset = prepare_dataset(self.tokenizer, dataset_config=self.config['dataset_config']['dataset_list'], max_seq_len=self.config['max_seq_len'])

        # Split Dataset into Train and Validation
        split_datasets = dataset.train_test_split(test_size=self.config['dataset_config'].get('test_size', 0.1), 
                                                  seed=self.config.get('random_seed', 42))
        train_dataset = split_datasets['train']
        val_dataset = split_datasets['test']
        self.accelerator.print(f"Train dataset size: {len(train_dataset)}")
        self.accelerator.print(f"Validation dataset size: {len(val_dataset)}")

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config['train_config']['batch_size'],
            shuffle=True,
            num_workers=self.config['train_config'].get('num_workers', 4),
            pin_memory=True)

        self.valid_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config['train_config']['batch_size'],
            shuffle=False,
            num_workers=self.config['train_config'].get('num_workers', 4),
            pin_memory=True)

        self.network = BERT_Wrapper(get_pretrained_bert_model(self.config['pretrained_model_name']))
        self.model = DiffusionModel(self.network, self.tokenizer.mask_token_id)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['train_config']['learning_rate']
        )

        (self.model, 
         self.optimizer, 
         self.train_dataloader,
         self.valid_dataloader) = self.accelerator.prepare(
            self.model, 
            self.optimizer, 
            self.train_dataloader,
            self.valid_dataloader
        )

    def train(self, save_path:str):
        os.makedirs(save_path, exist_ok=True)

        for epoch in range(self.start_epoch, self.config['train_config']['num_epochs'] + 1):
            self.model.train()
            epoch_start_time = time.time()
            total_loss = 0.0
            
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{self.config['train_config']['num_epochs']}")
            for step, batch in enumerate(progress_bar):
                with self.accelerator.accumulate(self.model):
                    x = batch['input_ids']
                    attention_mask = batch['attention_mask']

                    t, noisy_x, mask_indices = self.model.forward_process(x)
                    loss = self.model.loss(x, t, noisy_x, mask_indices, attention_mask)
                    
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    total_loss += loss.item()
                    progress_bar.set_postfix({'Train Loss': total_loss / (step + 1)})
                self.global_step += 1

                # Evaluate and Save Checkpoint
                if self.global_step % self.config['train_config'].get('eval_steps', 1000) == 0:
                    valid_loss = self.evaluate()
                    self.model.train()
                    self.accelerator.print(f"Step {self.global_step} | Valid Loss: {valid_loss:.4f}")
                    if self.best_valid_loss > valid_loss:
                        self.best_valid_loss = valid_loss
                        self.save_checkpoint(save_path, epoch, self.global_step, self.best_valid_loss)
                        self.accelerator.print(f"New best model saved with Valid Loss: {self.best_valid_loss:.4f}")

            # End of Epoch Evaluation
            valid_loss = self.evaluate()
            epoch_time = time.time() - epoch_start_time
            self.accelerator.print(f"Epoch {epoch} Done | Time: {epoch_time:.1f}s | Train Loss: {total_loss / len(self.train_dataloader):.4f} | Valid Loss: {valid_loss:.4f}")
            if self.best_valid_loss > valid_loss:
                self.best_valid_loss = valid_loss
                self.save_checkpoint(save_path, epoch + 1, self.global_step, self.best_valid_loss)
                self.accelerator.print(f"New best model saved with Valid Loss: {self.best_valid_loss:.4f}")

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        
        for batch in self.valid_dataloader:
            x = batch['input_ids']
            attention_mask = batch['attention_mask']

            t, noisy_x, mask_indices = self.model.forward_process(x)
            loss = self.model.loss(x, t, noisy_x, mask_indices, attention_mask)
            total_loss += loss.item()
        
        return total_loss / len(self.valid_dataloader)

    def save_checkpoint(self, save_path, epoch, steps, valid_loss):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        checkpoint = {
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'steps': steps,
            'valid_loss': valid_loss
        }
        
        file_name = f"epoch-{epoch}.pt"
        path = os.path.join(save_path, file_name)
        
        self.accelerator.save(checkpoint, path)
        self.accelerator.print(f"💾 Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.accelerator.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.global_step = checkpoint['steps']
            self.best_valid_loss = checkpoint['valid_loss']
            self.accelerator.print(f"✅ Loaded checkpoint from {path} (Epoch {self.start_epoch}, Step {self.global_step})")
            self.accelerator.print(f"    Best Validation Loss: {self.best_valid_loss:.4f}")
        except Exception as e:
            self.accelerator.print(f"❌ Failed to load checkpoint from {path}: {e}")