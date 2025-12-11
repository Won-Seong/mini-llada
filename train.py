import os
import yaml
import argparse
import torch
from transformers import (
    Trainer, 
    TrainingArguments,
    AutoTokenizer,
    AutoModel, 
    DataCollatorWithPadding
)

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
    parser.add_argument("--model_name", type=str, default="JuyeopDang/KoMiniLLaDA-0.7B-Base", help="Model name or path.")
    parser.add_argument("--config_file", type=str, default="configs/config.yaml", help="Path to the config file.")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--mode", type=str, default="pretrain", choices=["pretrain", "sft"], help="Training mode: pretrain or sft.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--from_scratch", action="store_true", help="Whether to train the model from scratch.")
    return parser

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = get_parser()
    args_cli = parser.parse_args()

    # 1. Load config file
    with open(args_cli.config_file, "r") as f:
        config = yaml.safe_load(f)

    # 2. Load tokenizer and model
    if args_cli.from_scratch:
        print("⚠️ Training from scratch. Initializing new model and tokenizer.")
        # 1. initialize tokenizer & config
        tokenizer = AutoTokenizer.from_pretrained(config['backbone_model_name'])
        
        llada_config = MiniLLaDAConfig(
            backbone_model_name=config['backbone_model_name'],
            mask_token_id=tokenizer.mask_token_id,
        )

        # 2. initialize model
        model = MiniLLaDA(llada_config)
        
        # 3. format for chat
        tokenizer = setup_chat_format(tokenizer)

        # Set config
        MiniLLaDAConfig.register_for_auto_class()
        
        # register model class for auto_map
        MiniLLaDA.register_for_auto_class("AutoModel")
        
        print("✅ Custom classes registered with auto_map.")
    else:    
        try:
            # load if the model exists in the Hub
            tokenizer = AutoTokenizer.from_pretrained(args_cli.model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(args_cli.model_name, trust_remote_code=True)
        except Exception as e:
            print(f"⚠️ No model in Hub or Error loading. Creating a local model... ({e})")
            return 0

    # 3. prepare dataset
    full_dataset = prepare_dataset(
        tokenizer,
        dataset_config=config['dataset_config']['pretrain' if args_cli.mode == 'pretrain' else 'sft']['dataset_list'], 
        max_seq_len=config['max_seq_len'],
        mode=args_cli.mode
    )

    # 3-1. train/test split
    test_size = config['dataset_config'].get('pretrain' if args_cli.mode == 'pretrain' else 'sft').get('test_size', 0.01)
    split_datasets = full_dataset.train_test_split(test_size=test_size, seed=config.get('random_seed', 42))
    
    train_dataset = split_datasets['train']
    eval_dataset = split_datasets['test']

    # generate 'labels' for Trainer
    # print("Mapping labels...")
    # train_dataset = train_dataset.map(lambda x: {'labels': x['input_ids']})
    # eval_dataset = eval_dataset.map(lambda x: {'labels': x['input_ids']})

    # 4. Set TrainingArguments
    train_conf = config['train_config']
    
    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        overwrite_output_dir=True,
        
        # Training Parameters
        num_train_epochs=train_conf.get('num_epochs', 3),
        per_device_train_batch_size=train_conf.get('batch_size', 8),
        per_device_eval_batch_size=train_conf.get('batch_size', 8),
        gradient_accumulation_steps=train_conf.get('gradient_accumulation_steps', 2),
        learning_rate=float(train_conf.get('learning_rate', 1e-5)),
        weight_decay=0.01,
        
        # Evaluation & Saving
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=train_conf.get('eval_steps', 1000),
        save_steps=train_conf.get('eval_steps', 1000),
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        
        # Hardware
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=train_conf.get('num_workers', 4),
        
        # Custom Model Settings
        remove_unused_columns=False, 
        
        # Logging
        logging_steps=100,
        report_to="none", 
        run_name="mini-llada-run",

        # Hub
        push_to_hub=True,
        hub_model_id=args_cli.model_name
    )

    # 5. Init Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[GenerateSampleCallback(tokenizer)]
    )

    # 6. Train
    print("🚀 Start Training...")
    trainer.train(resume_from_checkpoint=args_cli.resume_from_checkpoint)

    # # 7. Save final model
    # print(f"💾 Saving final model to {args_cli.output_dir}/final")
    # trainer.save_model(os.path.join(args_cli.output_dir, "final"))
    # tokenizer.save_pretrained(os.path.join(args_cli.output_dir, "final"))

if __name__ == "__main__":
    main()