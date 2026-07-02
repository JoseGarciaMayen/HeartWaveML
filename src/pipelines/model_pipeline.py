import argparse

from clearml import PipelineController


def run_tune(model_name: str) -> str:
    import os
    import sys

    sys.path.insert(0, "/home/jose/HeartWaveML")
    os.chdir("/home/jose/HeartWaveML")
    from src.pipeline import tune

    tune(model_name)
    return model_name


def run_train(model_name: str) -> str:
    import os
    import sys

    sys.path.insert(0, "/home/jose/HeartWaveML")
    os.chdir("/home/jose/HeartWaveML")
    from src.pipeline import train

    train(model_name)
    return model_name


def run_evaluate(model_name: str) -> None:
    import os
    import sys

    sys.path.insert(0, "/home/jose/HeartWaveML")
    os.chdir("/home/jose/HeartWaveML")
    from src.pipeline import evaluate

    evaluate(model_name)


def build_pipeline(model_name: str) -> PipelineController:
    pipe = PipelineController(name=f"{model_name}_pipeline", project="HeartWaveML", version="1.0.0")
    pipe.set_default_execution_queue("default")
    pipe.add_parameter(name="model_name", default=model_name)
    pipe.add_function_step(
        name="tune",
        function=run_tune,
        function_kwargs={"model_name": "${pipeline.model_name}"},
        function_return=["model_name"],
        packages=False,
    )
    pipe.add_function_step(
        name="train",
        function=run_train,
        function_kwargs={"model_name": "${tune.model_name}"},
        function_return=["model_name"],
        parents=["tune"],
        packages=False,
    )
    pipe.add_function_step(
        name="evaluate",
        function=run_evaluate,
        function_kwargs={"model_name": "${train.model_name}"},
        parents=["train"],
        packages=False,
    )
    return pipe


if __name__ == "__main__":
    from src.pipeline import MODELS

    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list(MODELS))
    args = parser.parse_args()
    pipe = build_pipeline(args.model)
    pipe.start(queue="default")
