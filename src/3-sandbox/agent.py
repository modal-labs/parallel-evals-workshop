import asyncio
import os

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

    # Return outputs back into model's context window
    # Be careful of token count!
    return {"stdout": stdout, "stderr": stderr, "exit_code": exit_code}


async def run_chat_agent(prompt: str, sandbox: modal.Sandbox):
    async with AsyncOpenAI(
        base_url="https://modal-labs-vg-demo--ep-qwen3-8-27b-server.us-west.modal.direct/v1",
        api_key=os.environ["QWEN_API_KEY"],
    ) as client:
        # Configure agent, this time with modal.Sandbox as run context
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

        # The SDK passes this sandbox to its tools through RunContextWrapper.
        result = await Runner.run(agent, prompt, context=sandbox, max_turns=None)
        return result.final_output
