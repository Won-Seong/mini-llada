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
from ko_mini_llada.data.dataset import prepare_dataset

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
    with open(args_cli.config_file, "r") as f:
        config = yaml.safe_load(f)

    # 2. Load tokenizer and model
    if args_cli.from_scratch:
        print("⚠️ Training from scratch. Initializing new model and tokenizer.")
        # 1. initialize tokenizer & config
        tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set pad_token to eos_token: {tokenizer.pad_token_id}")

        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({'mask_token': '[MASK]'})
            print(f"Added [MASK] token: {tokenizer.mask_token_id}")

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

        # 2. initialize model
        model = MiniLLaDA(llada_config)
        
        # 3. format for chat
        tokenizer = setup_chat_format(tokenizer)

        max_seq_len = model_conf.get('max_seq_len', 2048)
        # Set config
        MiniLLaDAConfig.register_for_auto_class()
        
        # register model class for auto_map
        MiniLLaDA.register_for_auto_class("AutoModel")
        
        print("✅ Custom classes registered with auto_map.")
    else:
        try:
            # load if the model exists in the Hub
            tokenizer = AutoTokenizer.from_pretrained(args_cli.model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = MiniLLaDA.from_pretrained(args_cli.model_name)
            max_seq_len = model.config.max_seq_len
        except Exception as e:
            print(f"⚠️ No model in Hub or Error loading. Creating a local model... ({e})")
            return 0

    # 3. prepare dataset
    full_dataset = prepare_dataset(
        tokenizer=tokenizer,
        dataset_config=config['dataset_config']['pretrain' if args_cli.mode == 'pretrain' else 'sft']['dataset_list'],
        max_seq_len=max_seq_len,
        mode=args_cli.mode
    )

    # 3-1. train/test split
    test_size = config['dataset_config'].get('pretrain' if args_cli.mode == 'pretrain' else 'sft').get('test_size', 0.001)
    split_datasets = full_dataset.train_test_split(test_size=test_size, seed=config.get('random_seed', 42))
    
    train_dataset = split_datasets['train']
    eval_dataset = split_datasets['test']

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
        bf16=train_conf.get('bf16', True),
        fp16=train_conf.get('fp16', True),
        dataloader_num_workers=train_conf.get('num_workers', 4),
        
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