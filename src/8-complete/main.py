from time import perf_counter

import modal
from agent import run_task_solver_agent
from models import MODELS, ModelConfig
from report import generate_report
from sandbox import prepare_sandbox, score_task
from tasks import Task, load_tasks

app = modal.App("evals-workshop-8-task-downloader")

harness_image = (
    modal.Image.debian_slim()
    .uv_pip_install("openai-agents", "matplotlib", "seaborn")
    .add_local_python_source("agent", "models", "sandbox", "tasks", "report")
)

# Set Modal Secrets with:
# modal secret create --from-dotenv .env llm-secrets
secrets = [modal.Secret.from_name("llm-secrets")]


@app.function(secrets=secrets, image=harness_image, timeout=3600, scaledown_window=5)
async def run_task(task: Task, model: ModelConfig):
    # Boot sandbox for task
    sandbox = await prepare_sandbox(task, app)
    try:
        # Time agent completion
        started = perf_counter()
        usage = await run_task_solver_agent(task, sandbox, model)
        seconds = perf_counter() - started

        # Calculate cost
        cost = None
        if model.token_prices is not None:
            input_price, output_price = model.token_prices
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
        for model in MODELS:
            call = await run_task.spawn.aio(task, model)
            calls.append((task.name, model, call))

    # Collect results
    results = {task.name: [] for task in tasks}
    for task_name, model, call in calls:
        result = await call.get.aio()
        results[task_name].append(result)
        print(f"[{task_name}] [{model.model_name}] {result}")

    generate_report(results, [model.model_name for model in MODELS])
