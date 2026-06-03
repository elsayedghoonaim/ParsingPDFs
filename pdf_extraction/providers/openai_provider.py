import logging
import os
from typing import Optional
from .openai_compatible import OpenAICompatibleProvider

logger = logging.getLogger("pdf_extraction")


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI VLM provider supporting GPT-4o, GPT-4o-mini, etc."""

    _batch_supported = True

    def __init__(self, config):
        super().__init__(config)
        self.base_url = config.provider_base_url or "https://api.openai.com/v1"
        self.api_key = config.provider_api_key or os.environ.get("OPENAI_API_KEY")
        self.model = config.provider_model

        # Only raise error if we are using the default OpenAI endpoint and have no API key
        if not self.api_key and "openai.com" in self.base_url:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or provider.api_key in config."
            )

    def estimate_cost(self, num_images: int, dpi: int) -> Optional[float]:
        """Estimates cost based on model and number of images/pages."""
        if self.config.cost_price_per_image is not None:
            return num_images * self.config.cost_price_per_image

        model_lower = self.model.lower()
        if "gpt-4o-mini" in model_lower:
            price = 0.001  # $0.001 per image
        elif "gpt-4o" in model_lower:
            price = 0.003  # $0.003 per image
        else:
            price = 0.003  # Fallback standard rate
        return num_images * price
