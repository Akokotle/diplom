# Contributing Guidelines

## Quality Control Tools

Automated tools are used to control code quality. All configuration details are managed via `.pre-commit-config.yaml` and `pyproject.toml`.

* **Formatter and Linter**: [Ruff](https://docs.astral.sh/ruff/)
* **Type Checker**: [Mypy](https://mypy-lang.org/)
* **Hooks Manager**: [Pre-Commit](https://pre-commit.com/)

## Environment Setup using uv

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
    # Installs hooks for pre-commit stage
    pre-commit install
    # Installs hooks for pre-push stage
    pre-commit install --hook-type pre-push
    ```



## Submitting Changes

1. **Sync main**
    Ensure your local main branch is up-to-date.
    ```bash
    git checkout main
    git pull origin main
    ```

2. **Create a new branch**
    Use a descriptive name
    ```bash
    git checkout -b name_of_your_branch
    ```

3. **Committing**
    ```bash
    # Stage your changes
    git add files_you_want_to_commit
    git commit -m "Description of your commit."
    ```

4. Upload your branch to GitHub
    ```bash
    git push -u origin name_of_your_branch
    ```

## How to Use Linters

### Automatic Checks (Pre-Commit Hooks)

All checks (Ruff Formatter, Ruff Linter, Mypy Type Checker) run automatically every time you execute `git commit`.

* **Automatic Fixing**: Ruff will automatically fix most formatting and style errors. If files were modified by the hook, you must `git add` those fixed files and commit again.
* **Commit Stoppage**: If Mypy detects typing errors or Ruff finds unfixable linting issues, the commit will be rejected.

### Manual Linter Usage

You can run all checks manually to verify the entire project state:

```bush
# Run all pre-commit hooks on all files (includes automatic Ruff fixing)
pre-commit run --all-files
```
