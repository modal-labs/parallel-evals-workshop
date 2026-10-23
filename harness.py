import modal
import asyncio
import math
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from modal.exception import SandboxFilesystemNotFoundError
from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunConfig,
    RunContextWrapper,
    Runner,
    function_tool,
)
from openai import AsyncOpenAI

import tasks
app = modal.App("evals-workshop")

harness_image = (
    modal.Image.debian_slim()
    .uv_pip_install("openai-agents", "harbor==0.24.0")
    .add_local_python_source("tasks")
)

# Set Modal Secrets with:
# modal secret create --from-dotenv .env llm-secrets
secrets=[modal.Secret.from_name("llm-secrets")]

@dataclass(frozen=True)
class EvalRunConfig:
    model_name: str
    base_url: str
    api_key_env: str

providers = [
    EvalRunConfig(
        model_name="gpt-4.1-mini",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    ),
    EvalRunConfig(
        model_name="moonshotai/Kimi-K3",
        base_url="https://modal-labs--ep-kimi-k3-internal-use-server.us-west.modal.direct/v1",
        api_key_env="KIMI_API_KEY",
    ),
]

@function_tool
def update_status(status: str):
    """Report a simple one-sentence status back to the user as you work"""
    print(status)


@function_tool
async def bash(context: RunContextWrapper[modal.Sandbox], command: str) -> dict:
    """Run a bash command in the sandbox and return stdout, stderr, and exit code.
    Files persist between calls, but each call starts a new shell.
    """
    print(f"CMD: {command}")
    sandbox = context.context
    process = await sandbox.exec.aio("bash", "-c", command, timeout=60)
    stdout, stderr = await asyncio.gather(
        process.stdout.read.aio(),
        process.stderr.read.aio(),
    )
    exit_code = await process.wait.aio()

    # If outputs are slim, return them verbatim
    output_bytes = len(stdout.encode("utf-8")) + len(stderr.encode("utf-8"))
    if output_bytes < 1024:
        return {"stdout": stdout, "stderr": stderr, "exit_code": exit_code}

    # Else write to file and return those instead
    prefix = f"/tmp/bash-output-{uuid.uuid4().hex}"
    stdout_path = prefix + ".stdout"
    stderr_path = prefix + ".stderr"
    await asyncio.gather(
        sandbox.filesystem.write_text.aio(stdout, stdout_path),
        sandbox.filesystem.write_text.aio(stderr, stderr_path),
    )
    return {
        "message": "Large output detected and written to files. Use shell tools (head, grep, etc) to selectively peek at the output:",
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "exit_code": exit_code,
    }




async def create_sandbox(env: tasks.Environment, timeout: int = 600):
    if env.config.get("docker_image"):
        image = modal.Image.from_registry(env.config["docker_image"])
    else:
        image = modal.Image.from_dockerfile(
            env.dockerfile,
            context_dir=env.context_dir,
        )

    return await modal.Sandbox.create.aio(
        app=app,
        image=image,
        timeout=timeout,
        workdir=env.config.get("workdir"),
    )


async def solve_task(task: tasks.PreparedTask, sandbox: modal.Sandbox, config: EvalRunConfig):
    agent = Agent[modal.Sandbox](
        name="Assistant",
        instructions="You are a helpful assistant. Use bash to work on your machine until the task is complete. Do not exit until complete.",
        tools=[bash],
        model_settings=ModelSettings(parallel_tool_calls=False),
    )
    async with AsyncOpenAI(
        base_url=config.base_url,
        api_key=os.environ[config.api_key_env],
    ) as client:
        model = OpenAIChatCompletionsModel(
            model=config.model_name,
            openai_client=client,
        )
        result = await Runner.run(
            agent,
            task.instruction,
            context=sandbox,
            run_config=RunConfig(model=model),
            max_turns=None,
        )
    return result.final_output


async def copy_artifacts(source: modal.Sandbox, target: modal.Sandbox, paths: list[str]):
    existing_paths = []
    for path in paths:
        try:
            await source.filesystem.stat.aio(path)
        except SandboxFilesystemNotFoundError:
            continue
        existing_paths.append(path.lstrip("/"))
    if not existing_paths:
        return

    archive = "/tmp/eval-artifacts.tar"
    process = await source.exec.aio("tar", "-cf", archive, "-C", "/", *existing_paths)
    stderr = await process.stderr.read.aio()
    if await process.wait.aio() != 0:
        raise RuntimeError(f"Artifact collection failed: {stderr}")

    with tempfile.TemporaryDirectory() as directory:
        local_archive = Path(directory) / "artifacts.tar"
        await source.filesystem.copy_to_local.aio(archive, local_archive)
        await target.filesystem.copy_from_local.aio(local_archive, archive)

    process = await target.exec.aio("tar", "-xf", archive, "-C", "/")
    stderr = await process.stderr.read.aio()
    if await process.wait.aio() != 0:
        raise RuntimeError(f"Artifact restoration failed: {stderr}")


async def score_task(task: tasks.PreparedTask, sandbox: modal.Sandbox):
    verifier = task.verifier
    timeout = math.ceil(verifier.config.get("timeout_sec", 600))
    grading_sandbox = sandbox
    if verifier.environment is not None:
        grading_sandbox = await create_sandbox(verifier.environment, timeout=timeout + 60)

    try:
        if grading_sandbox is not sandbox:
            await copy_artifacts(sandbox, grading_sandbox, verifier.artifacts)

        # Separate verifier images already contain their /tests directory.
        if verifier.environment is None or verifier.environment.context_dir == task.environment.context_dir:
            for path in verifier.tests_dir.rglob("*"):
                if path.is_file():
                    destination = "/tests/" + path.relative_to(verifier.tests_dir).as_posix()
                    await grading_sandbox.filesystem.copy_from_local.aio(path, destination)

        await grading_sandbox.filesystem.make_directory.aio("/logs/verifier")
        for filename in ("reward.json", "reward.txt"):
            try:
                await grading_sandbox.filesystem.remove.aio(f"/logs/verifier/{filename}")
            except SandboxFilesystemNotFoundError:
                pass

        process = await grading_sandbox.exec.aio(
            "bash", "/tests/test.sh",
            env=verifier.config.get("env", {}),
            timeout=timeout,
        )
        stdout, stderr = await asyncio.gather(
            process.stdout.read.aio(),
            process.stderr.read.aio(),
        )
        exit_code = await process.wait.aio()

        # A failed test may exit nonzero and still write a valid zero reward.
        for filename in ("reward.json", "reward.txt"):
            try:
                contents = await grading_sandbox.filesystem.read_text.aio(f"/logs/verifier/{filename}")
            except SandboxFilesystemNotFoundError:
                continue
            return tasks.parse_reward(contents)
        raise RuntimeError(f"Verifier produced no reward (exit {exit_code}):\n{stdout}\n{stderr}")
    finally:
        if grading_sandbox is not sandbox:
            await grading_sandbox.terminate.aio()


@app.function(secrets=secrets, image=harness_image, timeout=3600)
async def run_task(task_name: str, config: EvalRunConfig):
    task = await tasks.prepare(task_name)
    sandbox = await create_sandbox(task.environment)
    try:
        await solve_task(task, sandbox, config)
        return await score_task(task, sandbox)
    finally:
        await sandbox.terminate.aio()

# Run an ephemeral sweep of every task and provider
@app.local_entrypoint()
async def main():
    task_names = await tasks.list_tasks()
    calls = []

    # trim down for dev
    task_names = task_names[0:1]

    # Spawn every task/model pair before collecting results
    for task_name in task_names:
        for config in providers:
            call = await run_task.spawn.aio(task_name, config)
            calls.append((task_name, config, call))

    # Collect results
    for task_name, config, call in calls:
        response = await call.get.aio()
        print(f"[{task_name}] [{config.model_name}] {response}")
