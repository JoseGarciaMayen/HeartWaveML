import argparse
import importlib

MODELS = {
    "xgb": {
        "tuner": ("src.tuning.tune_xgb", "TunerXGB"),
        "trainer": ("src.training.train_xgb", "TrainerXGB"),
    },
    "convxgb": {
        "tuner": ("src.tuning.tune_convxgb", "TunerCONVXGB"),
        "trainer": ("src.training.train_convxgb", "TrainerCONVXGB"),
    },
    "cnn_mlp": {
        "tuner": ("src.tuning.tune_cnn_mlp", "TunerCNNMLP"),
        "trainer": ("src.training.train_cnn_mlp", "TrainerCNNMLP"),
        "tune_kwargs": lambda: {
            "pruner": importlib.import_module("optuna").pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=10
            )
        },
    },
    "extratrees": {
        "tuner": ("src.tuning.tune_extratrees", "TunerExtraTrees"),
        "trainer": ("src.training.train_extratrees", "TrainerExtraTrees"),
    },
    "lgbm": {
        "tuner": ("src.tuning.tune_lgbm", "TunerLGBM"),
        "trainer": ("src.training.train_lgbm", "TrainerLGBM"),
    },
    "catboost": {
        "tuner": ("src.tuning.tune_catboost", "TunerCatBoost"),
        "trainer": ("src.training.train_catboost", "TrainerCatBoost"),
    },
    "ensemble": {
        "trainer": ("src.training.train_ensemble", "TrainerEnsemble"),
    },
}


def _load_class(module_path: str, class_name: str):
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def tune(model_key: str):
    cfg = MODELS[model_key]
    if "tuner" not in cfg:
        raise SystemExit(
            f"'{model_key}' has no tuner. Configure its settings in config.yaml instead."
        )
    cls = _load_class(*cfg["tuner"])
    kwargs = cfg["tune_kwargs"]() if "tune_kwargs" in cfg else {}
    print(f"\n--- Tuning {model_key} ---")
    cls().run(**kwargs)


def train(model_key: str):
    cfg = MODELS[model_key]
    cls = _load_class(*cfg["trainer"])
    print(f"\n--- Training {model_key} ---")
    cls().train()


def run(model_key: str):
    tune(model_key)
    train(model_key)


def main():
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Tune and train ECG classification models.",
    )
    parser.add_argument("command", choices=["tune", "train", "run"])
    parser.add_argument(
        "model",
        help=f"Model to use: {', '.join(MODELS)} or 'all'",
    )
    args = parser.parse_args()

    if args.model != "all" and args.model not in MODELS:
        parser.error(f"Unknown model '{args.model}'. Choose from: {', '.join(MODELS)} or 'all'")

    keys = list(MODELS) if args.model == "all" else [args.model]
    action = {"tune": tune, "train": train, "run": run}[args.command]
    for key in keys:
        action(key)


if __name__ == "__main__":
    main()
