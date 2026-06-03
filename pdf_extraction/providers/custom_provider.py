import logging
import os
from typing import Optional
from .openai_compatible import OpenAICompatibleProvider

logger = logging.getLogger("pdf_extraction")


class CustomProvider(OpenAICompatibleProvider):
    """Custom VLM provider with user-defined base URL, headers, and batch capabilities."""

    def __init__(self, config):
        super().__init__(config)
        self.base_url = config.custom_base_url
        self.api_key = (
            config.custom_api_key
            or os.environ.get("CUSTOM_API_KEY")
            or config.provider_api_key
            or os.environ.get("PROVIDER_API_KEY")
            or ""
        )
        self.model = config.custom_model
        self.extra_headers = config.custom_headers or {}
        self._batch_supported = config.custom_supports_batch

    def estimate_cost(self, num_images: int, dpi: int) -> Optional[float]:
        """Custom endpoint estimation using the cost_price_per_image override if provided."""
        if self.config.cost_price_per_image is not None:
            return num_images * self.config.cost_price_per_image
        return None
