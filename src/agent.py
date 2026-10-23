import asyncio
import os
import uuid
from dataclasses import dataclass

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

from src.models import ModelConfig
from src.tasks import Task


# State passed into each tool
@dataclass
class RunContext:
    sandbox: modal.Sandbox
    model_name: str
    task_name: str


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
            context=RunContext(
                sandbox=sandbox, task_name=task.name, model_name=config.model_name
            ),
            run_config=RunConfig(model=model),
            max_turns=None,
        )
    return result.context_wrapper.usage
