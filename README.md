# img

Image processing.

## Repository structure

```
local_data/    large/local outputs (gitignored): res/, fig/
links/         curated symlinks to data inputs (tracked)
plan/          implementation plans
report/        analysis reports
docs/          documentation
env/           conda/pip environment snapshots
scripts/       standalone analysis scripts (+ common.py)
src/img/       main package
```

## Environment

This project uses a dedicated conda env named `img`. Run scripts with:

```bash
conda run -n img python -u <script>.py
# or, for long-running jobs with live output:
conda run --no-capture-output -n img python -u <script>.py
```

Install the package in editable mode:

```bash
conda run -n img pip install -e .
```
