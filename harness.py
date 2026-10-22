import modal
import uuid
from dataclasses import dataclass, field
from agents import Agent, Runner, Session
from agents.items import TResponseInputItem

app = modal.App("evals-workshop")

harness_image = modal.Image.debian_slim().uv_pip_install("openai-agents")
sandbox_image = modal.Image.debian_slim()

# Set Modal Secrets with:
# modal secret create --from-dotenv .env llm-secrets
secrets=[modal.Secret.from_name("llm-secrets")]

# Create a Modal Dict for persisting chat session history
sessions = modal.Dict.from_name("evals-workshop-sessions", create_if_missing=True)

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


        self.log(f"Created sandbox {self.sandbox.object_id}")

    @modal.exit()
    def exit(self):
        # Clean up sandbox
        self.sandbox.terminate()
        self.log(f"Terminated sandbox {self.sandbox.object_id}")


    # --------------------------
    # Custom interfaces we expose

    @modal.method()
    def chat(self, prompt):
        result = Runner.run_sync(self.agent, prompt, session=self.session)
        self.log(result.final_output)

    # --------------------------
    # Helpers

    def log(self, *args, **kwargs):
        print(f"[{self.id}]", *args, **kwargs)

def new_id():
    return str(uuid.uuid4())

@app.local_entrypoint()
def main(id: str | None = None, prompt="Write a haiku about programming"):
    if not id:
        id = new_id()

    harness = Harness(id=id)

    harness.chat.remote(prompt)
