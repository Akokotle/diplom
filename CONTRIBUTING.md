# Contributing Guidelines

Welcome! This is a GitHub repository where I will be documenting and saving my work related to my diploma thesis.

## Quality Control Tools

* **Formatter and Linter**: [Ruff](https://docs.astral.sh/ruff/)
* **Type Checker**: [Mypy](https://mypy-lang.org/)
* **Hooks Manager**: [Pre-Commit](https://pre-commit.com/)

## Environment Setup using uv

We use uv for dependency management and environment creation.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Akokotle/diplom.git
    cd diplom
    ```

2.  **Create and activate the virtual environment with uv:**
    ```bash
    # Create .venv
    uv venv --python 3.13
    # Activate
    source .venv/bin/activate
    ```

3.  **Install dependencies (core and dev) from pyproject.toml:**
    ```bash
    uv pip install -e ".[dev]"
    ```

4.  **Install pre-commit hooks:**
    ```bash
    pre-commit install --install-hooks
    pre-commit install --install-hooks --hook-type pre-push
    ```

## How to Use Linters

### Automatic Checks (Pre-Commit Hooks)

All checks (Ruff Formatter, Ruff Linter, Mypy Type Checker) run automatically every time you execute `git commit`.

* **Automatic Fixing**: Ruff will automatically fix most formatting and style errors. If files were modified by the hook, you must `git add` those fixed files and commit again.
* **Commit Stoppage**: If Mypy detects typing errors or Ruff finds unfixable linting issues, the commit will be rejected.

### Manual Linter Usage

You can run all checks manually to verify the entire project state, not just staged files:

```bash
# Run all pre-commit hooks on all files (includes automatic Ruff fixing)
pre-commit run --all-files
```
