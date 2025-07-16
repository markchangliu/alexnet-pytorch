import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(os.getcwd()).parent))

from src.train import run_train


def main() -> None:
    cfg_p = "train_cfg.json"
    run_train(cfg_p)


if __name__ == "__main__":
    main()