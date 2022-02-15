import wandb
from pathlib import Path
import subprocess

parser_opt_dir = Path("classification/options/parsers")

with wandb.init():
    config = wandb.config
    opt = config["opt"]
    opt_path = parser_opt_dir / opt
    print(f"TRAINING {opt}")
    subprocess.run(
        [
            "python",
            "classification/train_classifier.py",
            "-opt",
            opt_path,
            "--mode",
            "train",
        ]
    )
