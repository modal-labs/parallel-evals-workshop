import tempfile
from pathlib import Path

import modal

from src.tasks import Task


async def prepare_sandbox(task: Task, app: modal.App):
    with tempfile.TemporaryDirectory() as directory:
        # Copy task's dockerfile to disk
        dockerfile = Path(directory) / "Dockerfile"
        dockerfile.write_text(task.dockerfile)

        # Dynamically build a modal image from it
        image = modal.Image.from_dockerfile(dockerfile, context_dir=directory)

        # Create sandbox
        return await modal.Sandbox.create.aio(
            app=app,
            image=image,
            timeout=600,
        )


async def score_task(task: Task, sandbox: modal.Sandbox):
    await sandbox.filesystem.write_text.aio(task.test_script, "/tmp/score.sh")
    process = await sandbox.exec.aio("bash", "/tmp/score.sh", timeout=60)
    await process.wait.aio()
    return float(await sandbox.filesystem.read_text.aio("/logs/verifier/reward.txt"))
