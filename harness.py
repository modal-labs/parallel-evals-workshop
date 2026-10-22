import modal
import uuid
from agents import Agent, Runner

app = modal.App("evals-workshop")

harness_image = modal.Image.debian_slim().uv_pip_install("openai-agents")
sandbox_image = modal.Image.debian_slim()

# Set Modal Secrets with:
# modal secret create --from-dotenv .env llm-secrets
secrets=[modal.Secret.from_name("llm-secrets")]

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
        result = Runner.run_sync(self.agent, prompt)
        self.log(result.final_output)

    # --------------------------
    # Helpers

    def log(self, *args, **kwargs):
        print(f"[{self.id}]", *args, **kwargs)

def new_id():
    return str(uuid.uuid4())

@app.local_entrypoint()
def main():
    id = new_id()
    harness = Harness(id=id)

    prompt = "Write a haiku about programming"
    harness.chat.remote(prompt)
