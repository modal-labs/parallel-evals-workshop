"""Task loading and verification helpers; no sandbox or agent execution."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harbor.models.task.task import Task
from harbor.registry.client.package import PackageDatasetClient
from harbor.tasks.client import TaskClient

# Our dataset is terminal-bench, using harbor to download instructions, environments, and solutions
DATASET = "terminal-bench/terminal-bench"

@dataclass
class Environment:
    context_dir: Path
    config: dict[str, Any]

    @property
    def dockerfile(self) -> Path:
        return self.context_dir / "Dockerfile"


@dataclass
class Verifier:
    tests_dir: Path
    config: dict[str, Any]

@dataclass
class PreparedTask:
    name: str
    instruction: str
    environment: Environment
    verifier: Verifier
    timeout: float | None


async def prepare(task_name: str) -> PreparedTask:
    # Find task name in harbor's terminal bench dataset
    metadata = await PackageDatasetClient().get_dataset_metadata(DATASET)
    task_id = next(
        (task_id for task_id in metadata.task_ids if task_id.get_name() == task_name),
        None,
    )
    if task_id is None:
        raise ValueError(f"Task {task_name!r} is not in dataset {DATASET!r}")

    download = await TaskClient().download_tasks(task_ids=[task_id])
    task = Task(download.results[0].path)
    if task.has_steps:
        raise NotImplementedError("Multi-step tasks are not supported yet")

    return PreparedTask(
        name=task.name,
        instruction=task.instruction,
        environment=Environment(
            context_dir=task.paths.environment_dir,
            config=task.config.environment.model_dump(mode="json"),
        ),
        verifier=Verifier(
            tests_dir=task.paths.tests_dir,
            config=task.config.verifier.model_dump(mode="json"),
        ),
        timeout=task.config.agent.timeout_sec,
    )


def parse_reward(reward_contents: str) -> float:
    """Parse a verifier's /logs/verifier/reward.txt contents."""
    return float(reward_contents.strip())


async def list_tasks(dataset: str = DATASET) -> list[str]:
    metadata = await PackageDatasetClient().get_dataset_metadata(dataset)
    downloads = await TaskClient().download_tasks(task_ids=metadata.task_ids)
    names = []
    for task_id, download in zip(metadata.task_ids, downloads.results):
        task = Task(download.path)
        if task.has_steps:
            # Multistep tasks not supported
            continue
        names.append(task_id.get_name())

    return sorted(names)
