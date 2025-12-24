import os
import yaml
import argparse
import torch
from transformers import (
    Trainer, 
    TrainingArguments,
    AutoTokenizer,
    AutoModel, 
    DataCollatorForSeq2Seq
)
from huggingface_hub import login
from dotenv import load_dotenv

# model & config
from ko_mini_llada.models.configuration_mini_llada import MiniLLaDAConfig
from ko_mini_llada.models.modeling_mini_llada import MiniLLaDA
from ko_mini_llada.data_processing.local_dataset import prepare_pretrain_dataset

# dataset
from datasets import load_from_disk

# callbacks
from ko_mini_llada.utils.callbacks import GenerateSampleCallback

# helper
from ko_mini_llada.utils.helper import setup_chat_format


def get_parser():
    parser = argparse.ArgumentParser(description="Train KoMiniLlada Model")
    parser.add_argument("--model_name", type=str, default="JuyeopDang/KoMiniLLaDA-0.3B-Base", help="Model name or path.")
    parser.add_argument("--config_file", type=str, default="configs/config.yaml", help="Path to the config file.")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--mode", type=str, default="pretrain", choices=["pretrain", "sft"], help="Training mode: pretrain or sft.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--from_scratch", action="store_true", help="Whether to train the model from scratch.")
    parser.add_argument("--training_dataset_path", type=str, default=None, help="Path to the training dataset.")
    parser.add_argument("--validation_dataset_path", type=str, default=None, help="Path to the validation dataset.")
    return parser

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    load_dotenv()

    # Hugging Face Login
    if os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))

    parser = get_parser()
    args_cli = parser.parse_args()

    # 1. Load config file
    with open(args_cli.config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Logic Split: From Scratch vs Pretrained
    if args_cli.from_scratch:
        print("⚠️ Training from scratch.")
        
        # 3. Prepare Dataset & Train Tokenizer (if needed)      
        print("Preparing dataset and training tokenizer...")
        train_dataset, tokenizer = prepare_pretrain_dataset(
            tokenizer=None,
            config=config,
            path=args_cli.training_dataset_path
        )
        
        # Save tokenizer immediately
        if args_cli.output_dir:
            os.makedirs(args_cli.output_dir, exist_ok=True)
            tokenizer.save_pretrained(args_cli.output_dir + '/tokenizer/')
            print(f"New tokenizer saved to {args_cli.output_dir}")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set pad_token to eos_token: {tokenizer.pad_token_id}")

        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({'mask_token': '[MASK]'})
            print(f"Added [MASK] token: {tokenizer.mask_token_id}")

        # Initialize Model Config with new vocab size
        model_conf = config.get('model_config', {})
        llada_config = MiniLLaDAConfig(
            vocab_size=len(tokenizer),
            mask_token_id=tokenizer.mask_token_id,
            dim=model_conf.get('dim', 512),
            depth=model_conf.get('depth', 12),
            head=model_conf.get('head', 16),
            intermediate_size=model_conf.get('intermediate_size', 1024),
            max_seq_len=model_conf.get('max_seq_len', 2048)
        )

        # Initialize Model
        model = MiniLLaDA(llada_config)
        
        # Format for chat
        tokenizer = setup_chat_format(tokenizer)

        # Set config & Register
        MiniLLaDAConfig.register_for_auto_class()
        MiniLLaDA.register_for_auto_class("AutoModel")
        
        print("✅ Custom classes registered with auto_map.")
    else:
        # Load existing model and tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(args_cli.model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = MiniLLaDA.from_pretrained(args_cli.model_name)
        except Exception as e:
            print(f"⚠️ No model in Hub or Error loading. Creating a local model... ({e})")
            return 0
            
        print(f"Model size: {model.num_parameters() / 1e9:.2f}B")

        # Prepare Dataset using existing tokenizer
        dataset_cache_dir = args_cli.training_dataset_path + "/lm_dataset"
        if os.path.exists(dataset_cache_dir):
            print(f"Loading processed dataset from {dataset_cache_dir}...")
            train_dataset = load_from_disk(dataset_cache_dir)
        else:
            print("Preparing dataset...")
            # We pass the loaded tokenizer here
            train_dataset, _ = prepare_pretrain_dataset(
                tokenizer=tokenizer,
                config=config,
                path=args_cli.training_dataset_path
            )
            train_dataset.save_to_disk(dataset_cache_dir)
            print(f"Dataset saved to {dataset_cache_dir}.")

    print(f"Model size: {model.num_parameters() / 1e9:.2f}B")

    # Validation Dataset
    if args_cli.validation_dataset_path:
        dataset_cache_dir = args_cli.validation_dataset_path + "/lm_dataset"
        if os.path.exists(dataset_cache_dir):
            print(f"Loading processed dataset from {dataset_cache_dir}...")
            eval_dataset = load_from_disk(dataset_cache_dir)
        else:
            print("Preparing validation dataset...")
            # Reuse the tokenizer (whether new or loaded)
            eval_dataset, _ = prepare_pretrain_dataset(
                tokenizer=tokenizer,
                config=config,
                path=args_cli.validation_dataset_path
            )
            eval_dataset.save_to_disk(dataset_cache_dir)
            print(f"Dataset saved to {dataset_cache_dir}.")
    else:
        eval_dataset = None

    # 4. Set TrainingArguments
    train_conf = config['train_config']
    
    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        overwrite_output_dir=True,
        
        # Training Parameters
        num_train_epochs=train_conf.get('num_epochs', 3),
        per_device_train_batch_size=train_conf.get('batch_size', 8),
        per_device_eval_batch_size=train_conf.get('batch_size', 8),
        gradient_accumulation_steps=train_conf.get('gradient_accumulation_steps', 1),
        learning_rate=float(train_conf.get('learning_rate', 1e-5)),
        max_grad_norm=train_conf.get('max_grad_norm', 1.0),
        warmup_steps=train_conf.get('warmup_steps', 1000),
        weight_decay=train_conf.get('weight_decay', 0.01),
        
        # Evaluation & Saving
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=train_conf.get('eval_steps', 10000),
        save_steps=train_conf.get('eval_steps', 10000),
        save_total_limit=train_conf.get('save_total_limit', 1),
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        
        # Hardware
        deepspeed=train_conf.get('deepspeed', None),
        bf16=train_conf.get('bf16', True),
        fp16=train_conf.get('fp16', False),
        dataloader_num_workers=train_conf.get('num_workers', 4),
        gradient_checkpointing=train_conf.get('gradient_checkpointing', True),
        
        # Custom Model Settings
        remove_unused_columns=False,
        
        # Logging
        logging_steps=train_conf.get('logging_steps', 100),
        report_to=train_conf.get('report_to', "none"),
        run_name="mini-llada-run",

        # Hub
        push_to_hub=True,
        hub_model_id=args_cli.model_name,
        hub_strategy="end"
    )

    # 5. Init Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer),
        callbacks=[GenerateSampleCallback(tokenizer, mode=args_cli.mode)]  # custom callback for sample generation
    )

    # 6. Train
    print("🚀 Start Training...")
    trainer.train(resume_from_checkpoint=args_cli.resume_from_checkpoint)

    # 7. Save final model
    print(f"💾 Saving final model to {args_cli.output_dir}/final")
    trainer.save_model(os.path.join(args_cli.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args_cli.output_dir, "final_model"))

if __name__ == "__main__":
    main()