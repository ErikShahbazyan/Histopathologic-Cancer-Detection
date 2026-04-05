from pathlib import Path
import yaml
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_config():
    with open(PROJECT_ROOT / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    # resolve paths relative to project root
    for key, val in cfg["paths"].items():
        cfg["paths"][key] = PROJECT_ROOT / val

    # build checkpoint lookup
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    cfg["checkpoint_files"] = {}
    for entry in cfg["models"]:
        if entry["checkpoint"]:
            cfg["checkpoint_files"][entry["name"]] = ckpt_dir / entry["checkpoint"]

    # device
    cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # reproducibility
    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])

    return cfg

CFG = load_config()
