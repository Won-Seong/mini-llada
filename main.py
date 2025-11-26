import yaml
import argparse
import os

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
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    config = {}
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config)

    # Resume from checkpoint if provided
    if args.resume_path is not None:
        trainer.load_checkpoint(args.resume_path)

    # Start training
    trainer.train(save_path=args.output_dir)

if __name__ == "__main__":
    main()