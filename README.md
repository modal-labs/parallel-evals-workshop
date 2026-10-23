Run the eval sweep:

```sh
uv run modal run src/main.py
```

Add tasks under `src/tasks/<name>/` matching the format of `hello-world`:
- `instruction.md`
- `environment/Dockerfile`
- `tests/test.sh`.

Download Terminal-Bench 2.0 tasks into the gitignored `src/_tasks/` directory:

```sh
uv run download_tasks.py
# Or select task names:
uv run download_tasks.py fix-code-vulnerability
```
