from dataclasses import dataclass

from agents import OpenAIChatCompletionsModel, OpenAIResponsesModel


@dataclass
class ModelConfig:
    model_name: str
    base_url: str
    api_key_env: str
    model_class: type[OpenAIResponsesModel | OpenAIChatCompletionsModel]
    token_prices: tuple[float, float]  # in/out per million


MODELS = [
    ModelConfig(
        model_name="gpt-6-luna",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        model_class=OpenAIResponsesModel,
        token_prices=(0.10, 0.50),
    ),
    ModelConfig(
        model_name="claude-haiku-5-5",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        model_class=OpenAIChatCompletionsModel,
        token_prices=(0.10, 0.50),
    ),
    ModelConfig(
        model_name="Qwen/Qwen3.8-27B",
        base_url="https://modal-labs-vg-demo--ep-qwen3-8-27b-server.us-west.modal.direct/v1",
        api_key_env="QWEN_API_KEY",
        model_class=OpenAIChatCompletionsModel,
        token_prices=(0.04, 2.30),
    ),
]
