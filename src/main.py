# main.py
import yaml
import argparse
from typing import Text
from task.train import Trainer
from task.infer import Inference

def main(config_path: Text, mode: Text) -> None:
    # Load cấu hình YAML
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    if mode == 'train':
        print(" Training started...")
        trainer = Trainer(config)
        trainer.train()
        print(" Training complete")
    elif mode == 'infer':
        print(" Now evaluating on test data...")
        infer = Inference(config)
        infer.generate_captions()
        print(" Inference complete!")
    else:
        raise ValueError(f" Unknown mode: {mode}. Choose from 'train' or 'infer'.")

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True,
                             help="Path to config YAML file")
    args_parser.add_argument('--mode', dest='mode', choices=['train', 'infer'], required=True,
                             help="Run mode: train or infer")
    args = args_parser.parse_args()

    main(args.config, args.mode)
