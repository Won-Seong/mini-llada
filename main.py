import yaml
import argparse
import os
from mini_llada.data.dataset import get_tokenizer, prepare_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Mini LLaDA Trainer")
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./checkpoints", 
        help="Directory to save model checkpoints"
    )

    parser.add_argument(
        "--config_file", 
        type=str, 
        default="config.yaml", 
        help="Path to the configuration YAML file"
    )

    parser.add_argument(
        "--resume_path", 
        type=str, 
        default=None, 
        help="Path to the checkpoint file to resume from (e.g., ./checkpoints/mini_llada.pth)"
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["pretrain", "sft"],
        default="pretrain",
        help="Training mode: 'pretrain' for pre-training, 'sft' for supervised fine-tuning"
    )

    parser.add_argument(
        "--pad_with_eos",
        action="store_true",
        help="Whether to pad sequences with EOS token instead of standard padding",
        default=False
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    config = {}
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    tokenizer = get_tokenizer(config['pretrained_model_name'])
    if args.mode == 'pretrain':
        dataset_config = config['dataset_config']['pre_training']['dataset_list']
    else:  # args.mode == 'sft'
        dataset_config = config['dataset_config']['fine_tuning']['dataset_list']

    dataset = prepare_dataset(tokenizer, dataset_config=dataset_config, max_seq_len=config['max_seq_len'], mode=args.mode,
                              pad_with_eos=args.pad_with_eos)

    trainer = Trainer(config, tokenizer, dataset)

    # Resume from checkpoint if provided
    if args.resume_path is not None:
        trainer.load_checkpoint(args.resume_path)

    # Start training
    trainer.train(save_path=args.output_dir)

if __name__ == "__main__":
    main()