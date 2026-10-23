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
    # ModelConfig(
    #     model_name="gpt-6.1-sol",
    #     base_url="https://api.openai.com/v1",
    #     api_key_env="OPENAI_API_KEY",
    #     model_class=OpenAIResponsesModel,
    #     token_prices=(2.00, 10.00),
    # ),
    # ModelConfig(
    #     model_name="gpt-6-astra",
    #     base_url="https://api.openai.com/v1",
    #     api_key_env="OPENAI_API_KEY",
    #     model_class=OpenAIResponsesModel,
    #     token_prices=(10.00, 50.00),
    # ),
    ModelConfig(
        model_name="claude-haiku-5-5",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        model_class=OpenAIChatCompletionsModel,
        token_prices=(0.10, 0.50),
    ),
    # ModelConfig(
    #     model_name="claude-opus-5-5",
    #     base_url="https://api.anthropic.com/v1",
    #     api_key_env="ANTHROPIC_API_KEY",
    #     model_class=OpenAIChatCompletionsModel,
    #     token_prices=(4.00, 20.00),
    # ),
    # ModelConfig(
    #     model_name="claude-fable-5-1",
    #     base_url="https://api.anthropic.com/v1",
    #     api_key_env="ANTHROPIC_API_KEY",
    #     model_class=OpenAIChatCompletionsModel,
    #     token_prices=(10.00, 50.00),
    # ),
    ModelConfig(
        model_name="moonshotai/Kimi-K3",
        base_url="https://modal-labs--ep-kimi-k3-internal-use-server.us-west.modal.direct/v1",
        api_key_env="KIMI_API_KEY",
        model_class=OpenAIChatCompletionsModel,
        # Modal's OpenRouter rate as a proxy for self-hosted token spend.
        # https://openrouter.ai/moonshotai/kimi-k3 (2026-10-22)
        token_prices=(3.00, 15.00),
    ),
]
