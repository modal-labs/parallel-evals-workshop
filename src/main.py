from time import perf_counter

import modal

from src.agent import solve_task
from src.models import MODELS, ModelConfig
from src.report import generate_report
from src.sandbox import prepare_sandbox, score_task
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


@app.function(secrets=secrets, image=harness_image, timeout=3600, scaledown_window=5)
async def run_task(task: Task, config: ModelConfig):
    # Boot sandbox for task
    sandbox = await prepare_sandbox(task, app)
    try:
        # Time agent completion
        started = perf_counter()
        usage = await solve_task(task, sandbox, config)
        seconds = perf_counter() - started

        # Calculate cost
        cost = None
        if config.token_prices is not None:
            input_price, output_price = config.token_prices
            cost = (
                usage.input_tokens * input_price + usage.output_tokens * output_price
            ) / 1_000_000

        return {
            "score": await score_task(task, sandbox),
            "seconds": seconds,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "estimated_cost_usd": cost,
        }
    finally:
        await sandbox.terminate.aio()


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
