import asyncio
import os
import uuid
from dataclasses import dataclass
from time import perf_counter

import modal
from agents import (
    Agent,
    ModelSettings,
    RunConfig,
    RunContextWrapper,
    Runner,
    function_tool,
)
from openai import AsyncOpenAI

from src.models import MODELS, ModelConfig
from src.report import generate_report
from src.sandbox import prepare_task, score_task
from src.tasks import Task, load_tasks

app = modal.App("evals-workshop")

harness_image = (
    modal.Image.debian_slim()
    .uv_pip_install("openai-agents")
    .add_local_python_source("src")
)

# Set Modal Secrets with:
# modal secret create --from-dotenv .env llm-secrets
secrets = [modal.Secret.from_name("llm-secrets")]


# State passed into each tool
@dataclass
class RunContext:
    sandbox: modal.Sandbox
    model_name: str
    task_name: str


@app.local_entrypoint()
async def main():
    tasks = load_tasks()
    calls = []

    # Spawn every task/model pair before collecting results
    for task in tasks:
        for config in MODELS:
            call = await run_task.spawn.aio(task, config)
            calls.append((task.name, config, call))

    # Collect results
    results = {task.name: [] for task in tasks}
    for task_name, config, call in calls:
        result = await call.get.aio()
        results[task_name].append(result)
        print(f"[{task_name}] [{config.model_name}] {result}")

    generate_report(results, [config.model_name for config in MODELS])


@app.function(secrets=secrets, image=harness_image, timeout=3600, scaledown_window=5)
async def run_task(task: Task, config: ModelConfig):
    # Boot sandbox for task
    sandbox = await prepare_task(task, app)
    try:
        # Time agent completion
        started = perf_counter()
        usage = await solve_task(task, sandbox, config)
        seconds = perf_counter() - started

        # Calculate cost
        cost = None
        if config.token_prices is not None:
            input_price, output_price = config.token_prices
            cost = (usage.input_tokens * input_price + usage.output_tokens * output_price) / 1_000_000

        return {
            "score": await score_task(task, sandbox),
            "seconds": seconds,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "estimated_cost_usd": cost,
        }
    finally:
        await sandbox.terminate.aio()


async def solve_task(task: Task, sandbox: modal.Sandbox, config: ModelConfig):
    agent = Agent[RunContext](
        name="Assistant",
        instructions="You are a helpful assistant. Use bash to work on your machine until the task is complete. Do not exit until complete.",
        tools=[bash, update_status],
        model_settings=ModelSettings(parallel_tool_calls=False),
    )
    async with AsyncOpenAI(
        base_url=config.base_url,
        api_key=os.environ[config.api_key_env],
    ) as client:
        model = config.model_class(
            model=config.model_name,
            openai_client=client,
        )
        result = await Runner.run(
            agent,
            task.instruction,
            context=RunContext(sandbox=sandbox, task_name=task.name, model_name=config.model_name),
            run_config=RunConfig(model=model),
            max_turns=None,
        )
    return result.context_wrapper.usage


@function_tool
def update_status(ctx: RunContextWrapper[RunContext], status: str):
    """Report a simple one-sentence status back to the user as you work"""
    print(f"{ctx.context.model_name}\t{ctx.context.task_name}\t{status}")


@function_tool
async def bash(ctx: RunContextWrapper[RunContext], command: str) -> dict:
    """Run a bash command in the sandbox and return stdout, stderr, and exit code.
    Files persist between calls, but each call starts a new shell.
    """
    print(command)
    sandbox = ctx.context.sandbox
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
