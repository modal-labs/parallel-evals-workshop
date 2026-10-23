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
    for path in sorted((Path(__file__).parent / "tasks").iterdir()):
        tasks.append(Task(
            name=path.name,
            instruction=(path / "instruction.md").read_text(),
            dockerfile=(path / "environment/Dockerfile").read_text(),
            test_script=(path / "tests/test.sh").read_text(),
        ))
    return tasks
