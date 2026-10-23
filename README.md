# Evals workshop
In this repo we'll build an evaluation setup on Modal Endpoints, Functions, and Sandboxes, allowing us to evaluate self-hosted endpoints against public models in a massive parallel eval sweep.

### Structure
The repo `/src` contains working stages of this project, building up to `8-complete`. See `changelog.md` at each level for a description of what's added at each stage.

### Setup

Install dependencies once:

```sh
uv sync
```

### Running the final sweep 

```sh
# Run the complete eval sweep:
uv run modal run src/8-complete/main.py
```

Tasks are automatically picked up from `src/8-complete/tasks/<name>/`. You are free to add more, matching the format of the `hello-world` example:
- `instruction.md`
- `environment/Dockerfile`
- `tests/test.sh`.

#### Optional: add terminal bench tasks

Optionally download more task definitions (Terminal-Bench 2.0, via Harbor) into gitignored `src/8-complete/_tasks/` directory:

```sh
uv run src/8-complete/download_tasks.py
# Or select task names:
uv run src/8-complete/download_tasks.py fix-code-vulnerability
```
