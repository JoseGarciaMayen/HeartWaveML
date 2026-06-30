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

MLFLOW_URI = f"http://{os.getenv('IP', '127.0.0.1')}:5000"
DATA = _cfg["data"]
CNN_ARCH = _cfg["cnn_arch"]
ENSEMBLE = _cfg["ensemble"]
TUNING = _cfg["tuning"]
FEATURE_EXTRACTOR = _cfg["feature_extractor"]
SEQUENCES = _cfg.get("sequences", {})
