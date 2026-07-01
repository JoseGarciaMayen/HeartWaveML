import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_cfg_path = Path(__file__).parent.parent / "config.yaml"
with open(_cfg_path) as f:
    _cfg = yaml.safe_load(f)

DATA = _cfg["data"]
TUNING = _cfg["tuning"]
SEQUENCES = _cfg.get("sequences", {})
