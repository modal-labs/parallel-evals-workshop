"""Task loading and verification helpers; no sandbox or agent execution."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harbor.models.task.task import Task
from harbor.models.task.artifacts import normalize_artifact_entries
from harbor.registry.client.package import PackageDatasetClient
from harbor.tasks.client import TaskClient
from harbor.utils.env import resolve_env_vars

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
    environment: Environment | None = None
    artifacts: list[str] = field(default_factory=list)

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

    verifier_environment = None
    verifier_config = task.config.verifier.model_dump(mode="json")
    verifier_config["env"] = resolve_env_vars(task.config.verifier.env)
    if task.config.verifier.environment is not None:
        verifier_environment = Environment(
            context_dir=task.paths.tests_dir,
            config=task.config.verifier.environment.model_dump(mode="json"),
        )
    elif verifier_config["environment_mode"] == "separate":
        verifier_environment = Environment(
            context_dir=task.paths.environment_dir,
            config=task.config.environment.model_dump(mode="json"),
        )

    artifacts = ["/logs/artifacts"]
    for artifact in normalize_artifact_entries(task.config.artifacts):
        if artifact.service not in (None, "main") or artifact.exclude:
            raise NotImplementedError("Sidecar artifacts and artifact exclusions are not supported yet")
        if artifact.source not in artifacts:
            artifacts.append(artifact.source)

    return PreparedTask(
        name=task.name,
        instruction=task.instruction,
        environment=Environment(
            context_dir=task.paths.environment_dir,
            config=task.config.environment.model_dump(mode="json"),
        ),
        verifier=Verifier(
            tests_dir=task.paths.tests_dir,
            config=verifier_config,
            environment=verifier_environment,
            artifacts=artifacts,
        ),
        timeout=task.config.agent.timeout_sec,
    )


def parse_reward(reward_contents: str) -> float:
    """Parse reward.txt or reward.json into a numeric score."""
    reward = json.loads(reward_contents)
    if isinstance(reward, dict):
        reward = reward["reward"]
    score = float(reward)
    if not math.isfinite(score):
        raise ValueError("Verifier reward must be finite")
    return score


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
