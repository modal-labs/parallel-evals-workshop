from dataclasses import dataclass
from pathlib import Path


@dataclass
class Task:
    name: str
    instruction: str
    dockerfile: str
    test_script: str


def load_tasks() -> list[Task]:
    tasks = []
    for directory in ("tasks", "_tasks"):
        root = Path(__file__).parent / directory
        if not root.exists():
            continue
        for path in sorted(root.iterdir()):
            tasks.append(
                Task(
                    name=path.name,
                    instruction=(path / "instruction.md").read_text(),
                    dockerfile=(path / "environment/Dockerfile").read_text(),
                    test_script=(path / "tests/test.sh").read_text(),
                )
            )
    return tasks
