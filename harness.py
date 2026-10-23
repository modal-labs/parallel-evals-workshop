import modal
import os
import uuid
from dataclasses import dataclass
from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner, Session
from agents.items import TResponseInputItem
from openai import AsyncOpenAI



app = modal.App("evals-workshop")

harness_image = modal.Image.debian_slim().uv_pip_install("openai-agents")
sandbox_image = modal.Image.debian_slim()

# Set Modal Secrets with:
# modal secret create --from-dotenv .env llm-secrets
secrets=[modal.Secret.from_name("llm-secrets")]

# Create a Modal Dict for persisting chat session history
sessions = modal.Dict.from_name("evals-workshop-sessions", create_if_missing=True)

# Extend Session to back it with the Modal Dict
class ModalDictSession(Session):
    session_settings = None

    def __init__(self, session_id: str):
        self.session_id = session_id

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        items = await sessions.get.aio(self.session_id, [])
        if limit is None:
            return items
        return items[-limit:] if limit > 0 else []

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        items_ = await sessions.get.aio(self.session_id, [])
        items_.extend(items)
        await sessions.put.aio(self.session_id, items_)

    async def pop_item(self) -> TResponseInputItem | None:
        items = await sessions.get.aio(self.session_id, [])
        if not items:
            return None
        item = items.pop()
        await sessions.put.aio(self.session_id, items)
        return item

    async def clear_session(self) -> None:
        await sessions.pop.aio(self.session_id, None)

@dataclass(frozen=True)
class EvalRunConfig:
    model_name: str
    base_url: str
    api_key_env: str

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

# A parameterized Cls with max_containers=1 gives us a globally unique harness in a container
@app.cls(max_containers=1, secrets=secrets, image=harness_image)
class Harness:
    # Parameters create distinct autoscaling groups
    id: str = modal.parameter()

    # --------------------------
    # Container lifecycle methods

    @modal.enter()
    def init(self):
        # Each Harness owns a sandbox
        self.sandbox = modal.Sandbox.create(
            app=app,
            image=sandbox_image,
            timeout=600,
        )

        self.agent = Agent(name="Assistant", instructions="You are a helpful assistant")
        self.session = ModalDictSession(self.id)

    @modal.exit()
    def exit(self):
        # Clean up sandbox
        self.sandbox.terminate()

    # --------------------------
    # Custom interfaces we expose

    async def prepare_task(self, task_name: str):
        # TODO: self.task = await tasks.prepare(task_name)
        # TODO: Build the image and create self.sandbox from the task specification.
        # TODO: Move task sandbox creation out of init() above.
        raise NotImplementedError("Task preparation is not implemented yet")

    async def solve_task(self, config: EvalRunConfig):
        # TODO: Run the existing SDK agent using self.task.instruction and config.
        # TODO: Bind the bash tool to self.sandbox.
        raise NotImplementedError("Task solving is not implemented yet")

    async def score_task(self):
        # TODO: Upload verifier files, execute the verifier, and read its reward file.
        # TODO: return tasks.parse_reward(reward_contents)
        raise NotImplementedError("Task scoring is not implemented yet")

    @modal.method()
    async def run_task(self, task_name: str, config: EvalRunConfig):
        await self.prepare_task(task_name)
        try:
            await self.solve_task(config)
            return await self.score_task()
        finally:
            self.sandbox.terminate()

    @modal.method()
    async def chat(self, prompt: str, config: EvalRunConfig):
        async with AsyncOpenAI(
            base_url=config.base_url,
            api_key=os.environ[config.api_key_env],
        ) as client:
            # Target model specified by EvalRunConfig
            model = OpenAIChatCompletionsModel(
                model=config.model_name,
                openai_client=client,
            )

            # Run agent loop using Dict-bac
            result = await Runner.run(
                self.agent,
                prompt,
                session=self.session, # Pass in Dict-backed session
                run_config=RunConfig(model=model), # Pass in target endpoint
            )
        return(result.final_output)


def new_id():
    return str(uuid.uuid4())


# Run a stateful chat targeting an ID
@app.local_entrypoint()
def chat(
    id: str,
    prompt: str = "Write a haiku about programming",
):
    harness = Harness(id=id)
    response = harness.chat.remote(prompt, providers[0])
    print(response)

# Run an ephemeral sweep of all providers
@app.local_entrypoint()
def run_all():
    prompt = "Write a haiku about programming"
    calls = []

    # Spawn parallel runs per config
    for config in providers:
        id = new_id()
        harness = Harness(id=id)
        calls.append((config, harness.chat.spawn(prompt, config)))

    # Collect results
    for config, call in calls:
        response = call.get()
        print(f"[{config.model_name}] {response}")
