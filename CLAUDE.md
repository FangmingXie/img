# CLAUDE.md

## Repository Structure

```
img/
├── CLAUDE.md
├── README.md
├── LICENSE
├── pyproject.toml
├── local_data/            # local/large data (gitignored)
│   ├── res/                  # results (computed data - csv, parquet, npy, etc)
│   └── fig/                  # visualization (png, html, etc)
├── links/                 # curated symlinks to data inputs, organized by sub-project (tracked)
│   └── <dataset>/            # one folder per input dataset
├── plan/                  # plans organized by project
├── report/                # analysis reports organized by project
├── docs/                  # documentation
├── env/                   # conda/pip environment snapshots (tracked)
├── scripts/               # standalone analysis scripts, organized by projects
│   ├── common.py             # shared utilities for scripts
│   └── <task>/               # one folder per task/sub-project
└── src/
    └── img/
        ├── __init__.py
        ├── main.py
        └── utils.py
```

## Git Branches

- `main`: active development branch

## Git Configuration

- user.name: FangmingXie
- user.email: fmxie1993@gmail.com

## Environment

- Use this conda env to run this project: `img`, that means to run any python script with `conda run -n img`
- For heavy-lifting scripts, prefer running unbuffered at both levels so output (progress, logs) streams in real time: use `conda run --no-capture-output -n img python -u <script>.py`

## .gitignore Notes

- `local_data/` is gitignored (for large or local-only data files)
- `links/` is tracked (curated symlinks to data inputs)
- `env/` is tracked (environment snapshots saved before installing new packages)

## coding styles
- Define all file paths (input and output files) in the beginning of each script as much as possible. Capitalize the variables that store these file paths.

**Simplify Relentlessly**: Remove complexity aggressively - the simplest design that works is usually best

#### Fail-Fast, No Fallbacks
- **No Silent Fallbacks**: Code must fail immediately when expected conditions aren't met. Silent fallback behavior masks bugs and creates unpredictable systems.
- **Explicit Error Messages**: When something goes wrong, stop execution with clear error messages explaining what failed and what was expected.
- **Example**: `raise ValueError(f"Required model {model_name} not found")` instead of falling back to first available model.

### ⚠️ **IMPORTANT: Rewrite Project - Breaking Changes Encouraged**

**This package is a complete rewrite**, not an actively used codebase with external dependencies. This means:

- **Breaking changes are encouraged** when they follow best practices
- **No backward compatibility constraints** - optimize for clean architecture
- **Clean module organization** - each module has a single, clear purpose

This approach ensures the codebase remains maintainable and forces explicit dependencies that make the architecture clear to all developers.

## Claude Code Automation Rules
- When operating in Plan Mode, ALWAYS save the finalized implementation plan as a distinct markdown file under the `plan/` folder before concluding the turn.
- Never execute modifications while Plan Mode is toggled active.

## ⚠️ **IMPORTANT: Installing a package
- do not attempt to install new packages without asking permission explicitly.
- before installing anything new, save a copy of the current package list under `env/`.
- always try using conda first, and pip later.
