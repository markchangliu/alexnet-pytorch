import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(os.getcwd()).parent))

from src.test import run_test


def main() -> None:
    cfg_p = "/data2/cliu/workspaces/alexnet-pytorch/demos/test_cfg.json"

    # test with horizontal flips and five-crops
    run_test(cfg_p, True)

    # test with center crop
    run_test(cfg_p, False)