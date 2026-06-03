import logging
import os
from typing import Optional
from .openai_compatible import OpenAICompatibleProvider

logger = logging.getLogger("pdf_extraction")


class VLLMProvider(OpenAICompatibleProvider):
    """vLLM provider using OpenAI-compatible chat completion API."""

    _batch_supported = True

    def __init__(self, config):
        super().__init__(config)
        self.base_url = config.provider_base_url or "http://localhost:8000/v1"
        self.api_key = config.provider_api_key or os.environ.get("VLLM_API_KEY")
        self.model = config.provider_model

    def estimate_cost(self, num_images: int, dpi: int) -> Optional[float]:
        """Local self-hosted execution has no API costs."""
        return 0.0
