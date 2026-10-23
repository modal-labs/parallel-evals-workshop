from dataclasses import dataclass
from agents import OpenAIChatCompletionsModel, OpenAIResponsesModel

@dataclass
class ModelConfig:
    model_name: str
    base_url: str
    api_key_env: str
    model_class: type[OpenAIResponsesModel | OpenAIChatCompletionsModel]

MODELS = [
    ModelConfig(
        model_name="gpt-6-luna",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        model_class=OpenAIResponsesModel,
    ),
    ModelConfig(
        model_name="gpt-6.1-sol",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        model_class=OpenAIResponsesModel,
    ),
    ModelConfig(
        model_name="gpt-6-astra",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        model_class=OpenAIResponsesModel,
    ),
    ModelConfig(
        model_name="claude-haiku-5-5",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        model_class=OpenAIChatCompletionsModel,
    ),
    ModelConfig(
        model_name="claude-opus-5-5",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        model_class=OpenAIChatCompletionsModel,
    ),
    ModelConfig(
        model_name="claude-fable-5-1",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        model_class=OpenAIChatCompletionsModel,
    ),
    ModelConfig(
        model_name="moonshotai/Kimi-K3",
        base_url="https://modal-labs--ep-kimi-k3-internal-use-server.us-west.modal.direct/v1",
        api_key_env="KIMI_API_KEY",
        model_class=OpenAIChatCompletionsModel,
    ),
]
