import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
PROJECT_DEVICE = os.getenv("PROJECT_DEVICE", "")
GPU_ENABLED = (
    os.getenv("HEARTWAVEML_ENABLE_GPU", "").lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
    or PROJECT_DEVICE.lower() == "cuda"
)
if not GPU_ENABLED:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_cfg_path = Path(__file__).parent.parent / "config.yaml"
with open(_cfg_path) as f:
    _cfg = yaml.safe_load(f)

DATA = _cfg["data"]
TUNING = _cfg["tuning"]
SEQUENCES = _cfg.get("sequences", {})
HARDWARE = _cfg.get("hardware", {})
XGB_DEVICE = os.getenv(
    "HEARTWAVEML_XGB_DEVICE",
    "cuda" if PROJECT_DEVICE.lower() == "cuda" else HARDWARE.get("xgb_device", "cpu"),
)


def configure_tf_threads() -> None:
    """Pin TF CPU thread pools per config.yaml (intra_op_threads: null = os.cpu_count()).
    Must run before any op executes."""
    import tensorflow as tf

    intra = HARDWARE.get("intra_op_threads") or os.cpu_count()
    inter = HARDWARE.get("inter_op_threads", 1)
    tf.config.threading.set_intra_op_parallelism_threads(intra)
    tf.config.threading.set_inter_op_parallelism_threads(inter)
