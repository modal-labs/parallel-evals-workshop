import os

import modal
from agents import Agent, OpenAIChatCompletionsModel, Runner
from openai import AsyncOpenAI

app = modal.App("evals-workshop-2-agent")

# Now we add image definition, similar to dockerfile builds
image = modal.Image.debian_slim().uv_pip_install("openai-agents")

# modal secret create --from-dotenv .env llm-secrets
secrets = [modal.Secret.from_name("llm-secrets")]


# Our function decorator takes the custom image and secrets
@app.function(image=image, secrets=secrets)
async def chat(prompt: str):
    # Target our chat completions endpoint
    async with AsyncOpenAI(
        base_url="https://modal-labs-vg-demo--ep-qwen3-8-27b-server.us-west.modal.direct/v1",
        api_key=os.environ["QWEN_API_KEY"],
    ) as client:
        # Construct an agent with no tools, and run the prompt against it
        agent = Agent(
            name="Assistant",
            instructions="You are a helpful assistant.",
            model=OpenAIChatCompletionsModel(
                model="Qwen/Qwen3.8-27B", openai_client=client
            ),
        )
        result = await Runner.run(agent, prompt)
        return result.final_output


@app.local_entrypoint()
async def main(prompt: str = "Introduce yourself, including your model name."):
    print(await chat.remote.aio(prompt))
