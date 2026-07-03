# Distributed Tuning with ClearML

Beyond experiment tracking, `PipelineController` (`src/pipelines/model_pipeline.py`)
can run the full **tune → train → evaluate** flow for a model as jobs on a
[ClearML](https://clear.ml/) queue, picked up by one or more `clearml-agent`
workers. This lets you tune several models **in parallel on different
machines** - e.g. run `transformer` on one VM, `seq2seq` on another, and
`cnn_mlp` on your own laptop, all reporting to the same ClearML server.

Three agent workers running pipeline steps for different models in parallel,
as seen in the ClearML UI:

![ClearML workers running pipelines in parallel](../assets/clearml_pipelines.png)

## Single machine

1. Get a free [ClearML server account](https://app.clear.ml/) and run
   `clearml-agent init` to write your credentials to `~/clearml.conf`
   (or copy an existing one from another machine).
2. Start **two** agent workers on the `default` queue:
   ```bash
   clearml-agent daemon --queue default
   clearml-agent daemon --queue default
   ```
   You need two, not one: `pipe.start()` enqueues a controller task in
   addition to the `tune`/`train`/`evaluate` steps, and the controller
   occupies one worker for the whole run. With a single worker, the
   controller and the step it just enqueued deadlock - the controller waits
   for the step to start, but no worker is free to run it.
3. Enqueue a pipeline:
   ```bash
   python -m src.pipelines.model_pipeline transformer
   ```
   Steps land in their own project per model (`HeartWaveML/transformer`),
   separate from the pipeline's own bookkeeping (visible under the
   **Pipelines** tab in the ClearML UI, not the Projects tree).

## Multiple machines

Repeat this on each machine you want to contribute workers, using a
**dedicated queue name per machine** so jobs land where you intend instead
of on whichever machine happens to be free:

```bash
# 1. clone the repo, install deps (each machine needs its own copy of the
#    tracked data, so run this after cloning):
git clone https://github.com/JoseGarciaMayen/HeartWaveML.git
cd HeartWaveML
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt clearml-agent

# 2. copy ~/clearml.conf and .dvc/config.local from a machine that already
#    has them, then pull the tracked data (models, features, sequences):
dvc pull

# 3. start two workers on a queue unique to this machine (only needs
#    --create-queue the first time):
clearml-agent daemon --queue my_machine_queue --create-queue
clearml-agent daemon --queue my_machine_queue

# 4. enqueue, pointing at that queue:
python -m src.pipelines.model_pipeline seq2seq --queue my_machine_queue
```

If the repo lives at a path other than `~/HeartWaveML` on that machine,
export `HEARTWAVEML_REPO=/your/path` **before** starting the agents (child
task processes inherit it from the daemon's own environment).

**Keeping agents alive across SSH disconnects:** on modern Ubuntu, closing
your SSH session can kill "backgrounded" processes too (systemd cleans up
the whole session on logout). Run the agent inside `tmux` and detach
(`Ctrl+B`, `D`) instead of closing the terminal.

**ARM machines (e.g. Oracle Ampere, Apple Silicon Linux VMs):** `psutil`
needs to compile from source there, which needs `python3-dev` and
`build-essential` installed first (`sudo apt-get install python3-dev
build-essential`) - otherwise `pip`/`uv pip install` fails with a missing
`Python.h` error.

For the debugging story behind these, see
[`JOURNEY.md`](JOURNEY.md#6-building-a-real-clearml-pipeline-what-actually-broke).
