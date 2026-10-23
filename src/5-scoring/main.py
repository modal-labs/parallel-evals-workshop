import modal
from agent import run_task_solver_agent
from sandbox import prepare_sandbox, score_task
from tasks import Task, load_tasks

app = modal.App("evals-workshop-5-scoring")
image = (
    modal.Image.debian_slim()
    .uv_pip_install("openai-agents")
    .add_local_python_source("agent", "sandbox", "tasks")
)
secrets = [modal.Secret.from_name("llm-secrets")]


# We replace our chat remote function with one that executes a single task
@app.function(image=image, secrets=secrets)
async def run_task(task: Task):
    # Boot sandbox according to task
    sandbox = await prepare_sandbox(task, app)
    try:
        # Use agent to solve task in the sandbox
        await run_task_solver_agent(task, sandbox)

        return await score_task(task, sandbox)
    finally:
        await sandbox.terminate.aio()


@app.local_entrypoint()
async def main():
    calls = []
    for task in load_tasks():
        call = await run_task.spawn.aio(task)
        calls.append((task.name, call))

    for task_name, call in calls:
        score = await call.get.aio()
        print(f"[{task_name}] Score: {score}")
