# /// script
# dependencies = ["harbor==0.24.0"]
# ///

# This script uses Harbor to download Terminal Bench 2.0 tasks

import argparse
import asyncio
import shutil
from pathlib import Path

from harbor.models.task.task import Task
from harbor.registry.client import RegistryClientFactory
from harbor.tasks.client import TaskClient


async def main():
    parser = argparse.ArgumentParser(description="Download Terminal-Bench 2.0 tasks.")
    parser.add_argument("tasks", nargs="*", help="Task names; defaults to all tasks.")
    args = parser.parse_args()

    registry = RegistryClientFactory.create()
    metadata = await registry.get_dataset_metadata("terminal-bench@2.0")
    task_ids = []
    for task_id in metadata.task_ids:
        if not args.tasks or task_id.get_name() in args.tasks:
            task_ids.append(task_id)

    downloads = await TaskClient().download_tasks(task_ids)
    destination = Path(__file__).parent / "_tasks"
    for task_id, download in zip(task_ids, downloads.results):
        task = Task(download.path)
        verifier = task.config.verifier
        if (
            task.has_steps
            or verifier.environment is not None
            or verifier.environment_mode == "separate"
            or not (download.path / "environment/Dockerfile").is_file()
            or not (download.path / "tests/test.sh").is_file()
        ):
            print(f"Skipping {task_id.get_name()}: requires a different task lifecycle")
            continue

        target = destination / task_id.get_name()
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(download.path / "instruction.md", target / "instruction.md")
        shutil.copy2(download.path / "task.toml", target / "task.toml")
        for directory in ("environment", "tests"):
            shutil.copytree(
                download.path / directory, target / directory, dirs_exist_ok=True
            )
        print(f"Downloaded {task_id.get_name()}")


if __name__ == "__main__":
    asyncio.run(main())
