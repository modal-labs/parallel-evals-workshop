import io
import tarfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Task:
    name: str
    instruction: str
    environment: bytes
    tests: bytes


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
                    environment=bundle_directory(path / "environment"),
                    tests=bundle_directory(path / "tests"),
                )
            )
    return tasks


def bundle_directory(path: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        archive.add(path, arcname=".")
    return buffer.getvalue()
