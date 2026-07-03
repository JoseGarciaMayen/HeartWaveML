# Contributing to HeartWaveML

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

You'll also need a [Dagshub account](https://dagshub.com/) to `dvc pull` the
tracked data/models, and a [ClearML account](https://app.clear.ml/) if you
plan to run tuning (`clearml-agent init`).

## Before opening a PR

Run the same gate CI runs:

```bash
make ci   # ruff check + ruff format --check + pytest
```

`make ci` must pass locally before pushing. `.github/workflows/main.yml` runs
the identical lint + test steps (plus a Docker build) on every push/PR to
`main`.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/), subject line
only (no body): `feat: ...`, `fix: ...`, `chore: ...`, `docs: ...`. Look at
`git log` for examples already in this repo.

## Pull requests

- Branch off `main`, open the PR against `main`.
- Keep PRs scoped to one change.
- CI (`test` + `build` jobs) must be green before merging.
- If your change touches the data pipeline or feature schema, see
  "Invariants" below - these are checked by `tests/test_splitter.py` and
  `tests/test_utils.py`, but it's worth knowing them before you start.

## Invariants (do not break)

- **Inter-patient split**: never mix DS1 and DS2 records. CV is a fixed
  subset of DS1 (`{108, 205, 223}`), never DS2. See `src/data/splitter.py`.
- **RR/HRV features are always the last 10 columns** of the `feat` dataset.
- **Scaler is fit on train only**, then applied unchanged to CV and test.
- **`CENTER_IDX = WINDOW // 2`** for the sequence models - always derive
  this from the configured window size, never hardcode it.

See [`docs/JOURNEY.md`](docs/JOURNEY.md) for the reasoning and bugs these
invariants were written to prevent.

## Questions

Open an issue, or reach out at josegarciamayen@gmail.com.
