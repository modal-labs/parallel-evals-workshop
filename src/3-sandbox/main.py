import modal
from agent import run_chat_agent

app = modal.App("evals-workshop-3-sandbox")
image = (
    modal.Image.debian_slim()
    .uv_pip_install("openai-agents")
    .add_local_python_source("agent")
)
secrets = [modal.Secret.from_name("llm-secrets")]


@app.function(image=image, secrets=secrets)
async def chat(prompt: str):
    # Give each run its own machine, and clean up when it finishes.
    sandbox = await modal.Sandbox.create.aio(
        app=app, image=modal.Image.debian_slim(), timeout=600
    )
    try:
        return await run_chat_agent(prompt, sandbox)
    finally:
        await sandbox.terminate.aio()


@app.local_entrypoint()
async def main(prompt: str = "Write hello world to /hello.txt, then read it back."):
    print(await chat.remote.aio(prompt))
