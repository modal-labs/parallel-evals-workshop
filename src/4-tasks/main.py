import tempfile
from pathlib import Path

import modal
from agent import run_chat_agent
from tasks import Task, load_tasks

app = modal.App("evals-workshop-4-tasks")
image = (
    modal.Image.debian_slim()
    .uv_pip_install("openai-agents")
    .add_local_python_source("agent", "tasks")
)
secrets = [modal.Secret.from_name("llm-secrets")]


@app.function(image=image, secrets=secrets)
async def chat(task: Task):
    # Build the sandbox image from the task's Dockerfile.
    with tempfile.TemporaryDirectory() as directory:
        dockerfile = Path(directory) / "Dockerfile"
        dockerfile.write_text(task.dockerfile)
        sandbox_image = modal.Image.from_dockerfile(dockerfile, context_dir=directory)
        sandbox = await modal.Sandbox.create.aio(
            app=app, image=sandbox_image, timeout=600
        )
    try:
        return await run_chat_agent(task.instruction, sandbox)
    finally:
        await sandbox.terminate.aio()


@app.local_entrypoint()
async def main():
    task = load_tasks()[0]
    print(await chat.remote.aio(task))
