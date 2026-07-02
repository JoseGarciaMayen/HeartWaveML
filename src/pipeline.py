import argparse
import importlib

MODELS = {
    "cnn_mlp": {
        "tuner": ("src.tuning.tune_cnn_mlp", "TunerCNNMLP"),
        "trainer": ("src.training.train_cnn_mlp", "TrainerCNNMLP"),
        "tune_kwargs": lambda: {
            "pruner": importlib.import_module("optuna").pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=10
            )
        },
    },
    "lstm": {
        "tuner": ("src.tuning.tune_lstm", "TunerLSTM"),
        "trainer": ("src.training.train_lstm", "TrainerLSTM"),
    },
    "seq2seq": {
        "tuner": ("src.tuning.tune_seq2seq", "TunerSeq2Seq"),
        "trainer": ("src.training.train_seq2seq", "TrainerSeq2Seq"),
        "tune_kwargs": lambda: {
            "pruner": importlib.import_module("optuna").pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=10
            )
        },
    },
    "transformer": {
        "tuner": ("src.tuning.tune_transformer", "TunerTransformer"),
        "trainer": ("src.training.train_transformer", "TrainerTransformer"),
        "tune_kwargs": lambda: {
            "pruner": importlib.import_module("optuna").pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=10
            )
        },
    },
    "tcn": {
        "tuner": ("src.tuning.tune_tcn", "TunerTCN"),
        "trainer": ("src.training.train_tcn", "TrainerTCN"),
        "tune_kwargs": lambda: {
            "pruner": importlib.import_module("optuna").pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=10
            )
        },
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


def evaluate(model_key: str):
    import src.evaluate as ev

    print(f"\n--- Evaluating {model_key} ---")
    if model_key == "all":
        ev.evaluate_all()
    else:
        if model_key not in ev.EVAL_CONFIG:
            raise SystemExit(
                f"'{model_key}' not in EVAL_CONFIG. Choose from: {', '.join(ev.EVAL_CONFIG)}"
            )
        ev.evaluate_model(model_key)


def main():
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Tune, train, and evaluate ECG classification models.",
    )
    parser.add_argument("command", choices=["tune", "train", "run", "evaluate"])
    parser.add_argument(
        "model",
        help=f"Model to use: {', '.join(MODELS)} or 'all'",
    )
    args = parser.parse_args()

    if args.command == "evaluate":
        evaluate(args.model)
        return

    if args.model != "all" and args.model not in MODELS:
        parser.error(f"Unknown model '{args.model}'. Choose from: {', '.join(MODELS)} or 'all'")

    keys = list(MODELS) if args.model == "all" else [args.model]
    action = {"tune": tune, "train": train, "run": run}[args.command]
    for key in keys:
        action(key)


if __name__ == "__main__":
    main()
