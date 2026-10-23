import io
import tarfile
import tempfile
from pathlib import Path

import modal
from tasks import Task


async def prepare_sandbox(task: Task, app: modal.App):
    with tempfile.TemporaryDirectory() as directory:
        # Restore the full build context, including files referenced by COPY.
        with tarfile.open(fileobj=io.BytesIO(task.environment)) as archive:
            archive.extractall(directory, filter="data")
        dockerfile = Path(directory) / "Dockerfile"

        # Dynamically build a modal image from it
        image = modal.Image.from_dockerfile(dockerfile, context_dir=directory)

        # Create sandbox
        return await modal.Sandbox.create.aio(
            app=app,
            image=image,
            timeout=600,
        )


async def score_task(task: Task, sandbox: modal.Sandbox):
    # Upload tests only after the agent finishes.
    await sandbox.filesystem.write_bytes.aio(task.tests, "/tmp/tests.tar")
    process = await sandbox.exec.aio(
        "bash",
        "-c",
        "mkdir -p /tests /logs/verifier && tar -xf /tmp/tests.tar -C /tests && bash /tests/test.sh",
        timeout=60,
    )
    await process.wait.aio()
    return float(await sandbox.filesystem.read_text.aio("/logs/verifier/reward.txt"))
