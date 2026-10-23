import asyncio
import os
import uuid

import modal
from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    Runner,
    function_tool,
)
from openai import AsyncOpenAI
from tasks import Task


@function_tool
async def bash(ctx: RunContextWrapper[modal.Sandbox], command: str) -> dict:
    """Run a bash command. Files persist between calls; each starts a new shell."""

    # Sandbox is injected as a dependency via RunContextWrapper, unpack it:
    sandbox = ctx.context

    # Exec command as a bash script in sandbox
    print(command)
    process = await sandbox.exec.aio("bash", "-c", command, timeout=60)

    # Await process completion
    stdout, stderr = await asyncio.gather(
        process.stdout.read.aio(), process.stderr.read.aio()
    )
    exit_code = await process.wait.aio()

    # Gracefully handle large stdout/stderr outputs.

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


# We replace our chat agent with one that runs a single completion of a Task.
async def run_task_solver_agent(task: Task, sandbox: modal.Sandbox):
    async with AsyncOpenAI(
        base_url="https://modal-labs-vg-demo--ep-qwen3-8-27b-server.us-west.modal.direct/v1",
        api_key=os.environ["QWEN_API_KEY"],
    ) as client:
        agent = Agent[modal.Sandbox](
            name="Assistant",
            instructions="You are a helpful assistant. Use bash to complete the task.",
            model=OpenAIChatCompletionsModel(
                model="Qwen/Qwen3.8-27B", openai_client=client
            ),
            # Give it our bash tool to call
            tools=[bash],
            model_settings=ModelSettings(parallel_tool_calls=False),
        )

        result = await Runner.run(
            agent, task.instruction, context=sandbox, max_turns=None
        )
        return result.final_output
