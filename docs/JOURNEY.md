# Project Journey - decisions, mistakes, and lessons learned

This is a running log of how HeartWaveML got to its current state: what we
tried, what broke, what we fixed, and why. `README.md` shows the project as
it is today; this file explains how it got there.

## 1. The starting point: an honest benchmark

The first real decision was about evaluation, not modeling. ECG datasets are
easy to over-report on if you split beats randomly, because beats from the
same patient end up in both train and test, and the model can partly
memorize a patient's own heartbeat shape. We used the standard **inter-patient
split** (de Chazal DS1/DS2) instead: every patient's beats stay entirely in
one set. This gives lower numbers than a random split, but they are honest - they tell you how the model does on a patient it has never seen. This
choice shaped everything that came after: a model that looks great on a
random split can still fail badly here.

## 2. Tree models: a fast baseline, then dropped

XGBoost, LightGBM, ExtraTrees, CatBoost and an ensemble were tried first,
since tree models are cheap to train and a good sanity check. Once the
inter-patient split was fully enforced (no leakage anywhere in the data
pipeline), their honest scores plateaued around f1_macro 0.51–0.58. That is
well below what sequence models later reached, so the tree models were
removed from the repo entirely rather than kept as dead weight.

## 3. Complex models beat trees, consistently

After the tree ensemble was dropped, every neural architecture tried since
(from a simple CNN baseline up through recurrent and attention-based
sequence models) outperformed it on the honest inter-patient split, and
each architectural change moved the score more than any single
hyperparameter tweak. The clearest pattern: giving a model more temporal
context to reason over (rather than classifying one isolated beat) is what
consistently helped the hardest class, S, since it's diagnosed by rhythm
across beats rather than the shape of any single one.

## 4. Bugs found along the way (and what they taught us)

### Butterworth filter leaking into feature columns

`split_data` applied the same low-pass Butterworth filter meant for the
187 raw signal samples to the entire row of the `feat` dataset - which
also includes the 46 engineered features and RR intervals. Those columns
are not signal samples, so filtering them was meaningless and corrupted
them. This silently hurt every model trained on `feat` (mainly CNN-MLP)
until it was caught and fixed to only filter the `sample_*` columns.
Lesson: when you reuse a transform written for one data shape ("187
signal samples") on a wider table that grew extra columns over time,
double check the transform is still scoped correctly.

### Model promotion was leaking the test set

The original `evaluate_all()` promoted a newly trained candidate to
production whenever it beat the previous production model's test-set
score. That is a classic and easy-to-miss mistake: the test set is
supposed to be evaluated once and never used to make decisions, but here
it was silently being used as a selection criterion on every retrain.
This was fixed so that promotion decisions are made on the **CV split**
only; the test set is now evaluated once per run purely to report numbers,
and never influences which model ships.

## 5. Experiment tracking: MLflow → ClearML

The project originally tracked experiments with MLflow, then migrated to
ClearML. ClearML brought a `PipelineController`
that can orchestrate multi-step jobs (tune → train → evaluate) on a queue
that any number of machines can service. It also can be activated and remotely viewed through their app.
This made possible multi-machine remote tuning easier (see below).

## 7. Scaling tuning across machines

Once the pipeline was reliable on one machine, the natural next step was
running hyperparameter tuning for different models on different machines
in parallel, instead of one model at a time. That surfaced a few things
worth remembering for next time:

- **A ClearML queue must exist before an agent can listen on it.**
  Custom queue names (anything other than `default`) need
  `clearml-agent daemon --queue <name> --create-queue` the first time.
- **Every machine needs two agent workers on its queue, not one.**
  `PipelineController.start()` enqueues a controller task in addition to
  the tune/train/evaluate steps, and the controller occupies one worker for
  the whole run. With only one worker per queue, the controller and its own
  first step would deadlock - the controller waits for the step to start,
  but no worker is free to run it.
- **Background agent processes don't reliably survive an SSH disconnect**
  on modern Ubuntu, because systemd can kill a user's whole session
  (including "backgrounded" processes) on logout. Running the agent inside
  `tmux` (detach instead of closing the terminal) is the reliable fix; if
  that's still not enough, `loginctl enable-linger <user>` tells systemd not
  to clean up that user's processes on logout.
- **ARM machines need extra setup.** `psutil` (and potentially other
  packages) need to compile from source on aarch64, which needs
  `python3-dev`/`build-essential` installed first - plain `pip install`
  fails there with a missing `Python.h` error until those system packages
  are present.

See [`README.md`](../README.md#distributed-tuning-with-clearml) for the
concrete, reproducible steps to set this up.

## 8. Where things stand

The best model today is the Transformer at f1_macro 0.753, still short of
the project's 0.80 target. The recurring bottleneck across every
architecture tried so far is the S class, which is diagnosed by rhythm
(RR intervals) more than by beat shape - every model that improved on S
did so by giving the network more temporal context (bigger windows,
attention across the window), not by improving raw feature quality. That
pattern is the best lead for what to try next.
