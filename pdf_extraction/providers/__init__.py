"""Provider factory — returns the correct provider based on config."""

from .openai_provider import OpenAIProvider
from .google_provider import GoogleProvider
from .ollama_provider import OllamaProvider
from .vllm_provider import VLLMProvider
from .custom_provider import CustomProvider


def get_provider(config):
    """
    Factory function. Returns an instance of the correct provider based on PipelineConfig.
    Raises ValueError if provider name is unknown.
    """
    providers = {
        "openai": OpenAIProvider,
        "google": GoogleProvider,
        "ollama": OllamaProvider,
        "vllm": VLLMProvider,
        "custom": CustomProvider,
    }
    name = config.provider_name
    if name not in providers:
        raise ValueError(f"Unknown provider '{name}'. Must be one of: {list(providers.keys())}")
    return providers[name](config)
