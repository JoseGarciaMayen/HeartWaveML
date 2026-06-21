import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

_cfg_path = Path(__file__).parent.parent / "config.yaml"
with open(_cfg_path) as f:
    _cfg = yaml.safe_load(f)

MLFLOW_URI = f"http://{os.getenv('IP', '127.0.0.1')}:5000"
DATA = _cfg["data"]
CNN_ARCH = _cfg["cnn_arch"]
ENSEMBLE = _cfg["ensemble"]
