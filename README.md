Install dependencies once:

```sh
uv sync
```


Add tasks under `src/8-complete/tasks/<name>/` matching the format of `hello-world`:
- `instruction.md`
- `environment/Dockerfile`
- `tests/test.sh`.


```sh
# Run the complete eval sweep:
uv run modal run src/8-complete/main.py
```

Optionally download more task definitions (Terminal-Bench 2.0, via Harbor) into gitignored `src/8-complete/_tasks/` directory:

```sh
uv run src/8-complete/download_tasks.py
# Or select task names:
uv run src/8-complete/download_tasks.py fix-code-vulnerability
```
