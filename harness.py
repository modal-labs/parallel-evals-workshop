import modal
import asyncio
import os
import uuid
from dataclasses import dataclass
from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunConfig,
    RunContextWrapper,
    Runner,
    function_tool,
)
from openai import AsyncOpenAI



app = modal.App("evals-workshop")

harness_image = modal.Image.debian_slim().uv_pip_install("openai-agents")

# Set Modal Secrets with:
# modal secret create --from-dotenv .env llm-secrets
secrets=[modal.Secret.from_name("llm-secrets")]

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


# Each trial ID gets its own harness container.
@app.cls(max_containers=1, secrets=secrets, image=harness_image)
class Harness:
    # Parameters create distinct autoscaling groups
    id: str = modal.parameter()

    # --------------------------
    # Container lifecycle methods

    @modal.enter()
    def init(self):
        self.sandbox = None
        self.init_agent()

    @modal.exit()
    def exit(self):
        self.terminate_sandbox()

    def terminate_sandbox(self):
        if self.sandbox is not None:
            self.sandbox.terminate()
            self.sandbox = None

    # --------------------------
    # Agent

    def init_agent(self):
        self.agent = Agent(
            name="Assistant",
            instructions="You are a helpful assistant. Use bash to work in your sandbox.",
            tools=[self.bash],
            model_settings=ModelSettings(parallel_tool_calls=False),
        )

    @staticmethod # self passed in via RunContextWrapper
    @function_tool
    async def bash(context: RunContextWrapper, command: str) -> dict:
        """Run a bash command in the sandbox and return stdout, stderr, and exit code.
        Files persist between calls, but each call starts a new shell.
        """
        self = context.context
        process = await self.sandbox.exec.aio("bash", "-c", command, timeout=60)
        stdout, stderr = await asyncio.gather(
            process.stdout.read.aio(),
            process.stderr.read.aio(),
        )
        exit_code = await process.wait.aio()
        return {"stdout": stdout, "stderr": stderr, "exit_code": exit_code}

    # --------------------------
    # Custom interfaces we expose

    async def prepare_task(self, task_name: str):
        # TODO: self.task = await tasks.prepare(task_name)
        # TODO: Build the image and create self.sandbox from the task specification.
        raise NotImplementedError("Task preparation is not implemented yet")

    async def solve_task(self, config: EvalRunConfig):
        async with AsyncOpenAI(
            base_url=config.base_url,
            api_key=os.environ[config.api_key_env],
        ) as client:
            # Target model specified by EvalRunConfig
            model = OpenAIChatCompletionsModel(
                model=config.model_name,
                openai_client=client,
            )

            # The SDK keeps conversation history within this trial's agent loop.
            result = await Runner.run(
                self.agent,
                self.task.instruction,
                context=self, # Tools receive this harness through RunContextWrapper
                run_config=RunConfig(model=model), # Pass in target endpoint
            )
        return(result.final_output)

    async def score_task(self):
        # TODO: Upload verifier files, execute the verifier, and read its reward file.
        # TODO: return tasks.parse_reward(reward_contents)
        raise NotImplementedError("Task scoring is not implemented yet")

    @modal.method()
    async def run_task(self, task_name: str, config: EvalRunConfig):
        try:
            await self.prepare_task(task_name)
            await self.solve_task(config)
            return await self.score_task()
        finally:
            self.terminate_sandbox()

def new_id():
    return str(uuid.uuid4())

# Run an ephemeral sweep of every task and provider
@app.local_entrypoint()
async def main():
    from tasks import list_tasks

    task_names = await list_tasks()
    calls = []

    # Spawn every task/model pair before collecting results
    for task_name in task_names:
        for config in providers:
            id = new_id()
            harness = Harness(id=id)
            call = await harness.run_task.spawn.aio(task_name, config)
            calls.append((task_name, config, call))

    # Collect results
    for task_name, config, call in calls:
        response = await call.get.aio()
        print(f"[{task_name}] [{config.model_name}] {response}")
