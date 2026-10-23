Run the workshop sweep:

```sh
uv run modal run src/main.py
```

Add tasks under `src/tasks/<name>/` with `instruction.md`, `environment/Dockerfile`, and `tests/test.sh`.

Download Terminal-Bench 2.0 tasks into the gitignored `src/_tasks/` directory:

```sh
uv run download_tasks.py
# Or select task names:
uv run download_tasks.py fix-code-vulnerability
```

The sweep loads both directories. Downloads preserve environment and test assets and omit solutions; multi-step tasks and separate verifiers are skipped.
The runner currently reads only the Dockerfile and test.sh; using supporting files in downloaded tasks needs runner support.
