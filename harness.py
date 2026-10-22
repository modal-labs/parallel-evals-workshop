import modal
import uuid

app = modal.App("evals-workshop")

# All sandboxes start on debian slim
sandbox_image = modal.Image.debian_slim()

# A parameterized Cls with max_containers=1 gives us a globally unique harness in a container
@app.cls(max_containers=1)
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
        self.log(prompt)

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

    prompt = "hello"
    harness.chat.remote(prompt)
