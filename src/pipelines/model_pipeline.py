import argparse
from pathlib import Path

from clearml import PipelineController, Task

REQUIREMENTS_PATH = Path(__file__).resolve().parent.parent.parent / "requirements.txt"

Task.force_requirements_env_freeze(requirements_file=str(REQUIREMENTS_PATH))


def _read_requirements() -> list[str]:
    lines = REQUIREMENTS_PATH.read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def run_tune(model_name: str) -> str:
    import os
    import sys

    repo_path = os.environ.get("HEARTWAVEML_REPO", os.path.expanduser("~/HeartWaveML"))
    sys.path.insert(0, repo_path)
    os.chdir(repo_path)
    from src.pipeline import tune

    tune(model_name)
    return model_name


def run_train(model_name: str) -> str:
    import os
    import sys

    repo_path = os.environ.get("HEARTWAVEML_REPO", os.path.expanduser("~/HeartWaveML"))
    sys.path.insert(0, repo_path)
    os.chdir(repo_path)
    from src.pipeline import train

    train(model_name)
    return model_name


def run_evaluate(model_name: str) -> None:
    import os
    import sys

    repo_path = os.environ.get("HEARTWAVEML_REPO", os.path.expanduser("~/HeartWaveML"))
    sys.path.insert(0, repo_path)
    os.chdir(repo_path)
    from src.pipeline import evaluate

    evaluate(model_name)


def build_pipeline(model_name: str, queue: str = "default") -> PipelineController:
    packages = _read_requirements()
    step_project = f"HeartWaveML/{model_name}"
    pipe = PipelineController(
        name=f"{model_name}_pipeline",
        project="HeartWaveML",
        version="1.0.1",
        target_project=False,
    )
    pipe.set_default_execution_queue(queue)
    pipe.add_parameter(name="model_name", default=model_name)
    pipe.add_function_step(
        name="tune",
        function=run_tune,
        function_kwargs={"model_name": "${pipeline.model_name}"},
        function_return=["model_name"],
        packages=packages,
        project_name=step_project,
    )
    pipe.add_function_step(
        name="train",
        function=run_train,
        function_kwargs={"model_name": "${tune.model_name}"},
        function_return=["model_name"],
        parents=["tune"],
        packages=packages,
        project_name=step_project,
    )
    pipe.add_function_step(
        name="evaluate",
        function=run_evaluate,
        function_kwargs={"model_name": "${train.model_name}"},
        parents=["train"],
        packages=packages,
        project_name=step_project,
    )
    return pipe


if __name__ == "__main__":
    from src.pipeline import MODELS

    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list(MODELS))
    parser.add_argument("--queue", default="default")
    args = parser.parse_args()
    pipe = build_pipeline(args.model, queue=args.queue)
    pipe.start(queue=args.queue)
