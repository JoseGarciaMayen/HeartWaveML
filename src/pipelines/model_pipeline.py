import argparse

from clearml import PipelineController


def run_tune(model_name: str) -> str:
    from src.pipeline import tune

    tune(model_name)
    return model_name


def run_train(model_name: str) -> str:
    from src.pipeline import train

    train(model_name)
    return model_name


def run_evaluate(model_name: str) -> None:
    from src.pipeline import evaluate

    evaluate(model_name)


def build_pipeline(model_name: str) -> PipelineController:
    pipe = PipelineController(name=f"{model_name}_pipeline", project="HeartWaveML", version="1.0.0")
    pipe.add_parameter(name="model_name", default=model_name)
    pipe.add_function_step(
        name="tune",
        function=run_tune,
        function_kwargs={"model_name": "${pipeline.model_name}"},
        function_return=["model_name"],
        execution_queue="default",
    )
    pipe.add_function_step(
        name="train",
        function=run_train,
        function_kwargs={"model_name": "${tune.model_name}"},
        function_return=["model_name"],
        parents=["tune"],
        execution_queue="default",
    )
    pipe.add_function_step(
        name="evaluate",
        function=run_evaluate,
        function_kwargs={"model_name": "${train.model_name}"},
        parents=["train"],
        execution_queue="default",
    )
    return pipe


if __name__ == "__main__":
    from src.pipeline import MODELS

    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list(MODELS))
    args = parser.parse_args()
    pipe = build_pipeline(args.model)
    pipe.start(queue=None)
