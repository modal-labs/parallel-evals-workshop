import modal
import asyncio
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
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

from tasks import Task, load_tasks
app = modal.App("evals-workshop")

harness_image = (
    modal.Image.debian_slim()
    .uv_pip_install("openai-agents")
    .add_local_python_source("tasks")
)

# Set Modal Secrets with:
# modal secret create --from-dotenv .env llm-secrets
secrets=[modal.Secret.from_name("llm-secrets")]

@dataclass
class EvalRunConfig:
    model_name: str
    base_url: str
    api_key_env: str

# State passed into each tool
@dataclass
class RunContext:
    sandbox: modal.Sandbox
    model_name: str
    task_name: str

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
def update_status(ctx: RunContextWrapper[RunContext], status: str):
    """Report a simple one-sentence status back to the user as you work"""
    print(f"{ctx.context.model_name}\t{ctx.context.task_name}\t{status}")


@function_tool
async def bash(ctx: RunContextWrapper[RunContext], command: str) -> dict:
    """Run a bash command in the sandbox and return stdout, stderr, and exit code.
    Files persist between calls, but each call starts a new shell.
    """
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

async def prepare_task(task: Task):
    with tempfile.TemporaryDirectory() as directory:
        dockerfile = Path(directory) / "Dockerfile"
        dockerfile.write_text(task.dockerfile)
        image = modal.Image.from_dockerfile(dockerfile, context_dir=directory)
        return await modal.Sandbox.create.aio(
            app=app,
            image=image,
            timeout=600,
        )

async def solve_task(task: Task, sandbox: modal.Sandbox, config: EvalRunConfig):
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
        model = OpenAIChatCompletionsModel(
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
    return result.final_output


async def score_task(task: Task, sandbox: modal.Sandbox):
    await sandbox.filesystem.write_text.aio(task.test_script, "/tmp/score.sh")
    process = await sandbox.exec.aio("bash", "/tmp/score.sh", timeout=60)
    await process.wait.aio()
    return float(await sandbox.filesystem.read_text.aio("/logs/verifier/reward.txt"))


@app.function(secrets=secrets, image=harness_image, timeout=3600)
async def run_task(task: Task, config: EvalRunConfig):
    sandbox = await prepare_task(task)
    try:
        await solve_task(task, sandbox, config)
        return await score_task(task, sandbox)
    finally:
        await sandbox.terminate.aio()

# Run an ephemeral sweep of every task and provider
@app.local_entrypoint()
async def main():
    tasks = load_tasks()
    calls = []

    # Spawn every task/model pair before collecting results
    for task in tasks:
        for config in providers:
            call = await run_task.spawn.aio(task, config)
            calls.append((task.name, config, call))

    # Collect results
    scores = {task.name: [] for task in tasks}
    for task_name, config, call in calls:
        response = await call.get.aio()
        scores[task_name].append(response)
        print(f"[{task_name}] [{config.model_name}] {response}")

    plot_scores(scores)


def plot_scores(scores: dict[str, list[float]]):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white", font_scale=1.05)
    averages = [sum(column) / len(column) for column in zip(*scores.values())]
    fig, ax = plt.subplots(figsize=(max(6, 3 * len(providers)), 0.5 * len(scores) + 2.5))
    sns.heatmap(
        [averages, *scores.values()],
        annot=True, fmt=".2f", cmap="crest", vmin=0, vmax=1,
        linewidths=2, linecolor="white",
        xticklabels=[config.model_name.rsplit("/", 1)[-1] for config in providers],
        yticklabels=["Average", *scores],
        cbar_kws={"label": "Score", "ticks": [0, 0.5, 1]}, ax=ax,
    )
    ax.set_title("Agent evaluation scores", loc="left", weight="bold", pad=20)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, labelrotation=0)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.axhline(1, color="white", linewidth=5)
    fig.savefig("report.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("Saved report.png")
