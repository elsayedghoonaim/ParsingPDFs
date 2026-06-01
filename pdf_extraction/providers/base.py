import aiohttp
from abc import ABC, abstractmethod
from typing import Optional


class BaseProvider(ABC):
    """Abstract base class for all VLM/OCR providers."""

    def __init__(self, config):
        """
        Store config. Extract provider-specific fields:
        - self.base_url (from config or default)
        - self.api_key
        - self.model
        """
        self.config = config

    @abstractmethod
    async def send_image(
        self,
        image_bytes: bytes,
        prompt: str,
        session: aiohttp.ClientSession
    ) -> str:
        """Send a single image with prompt. Return extracted text."""
        pass

    @abstractmethod
    async def send_images_batch(
        self,
        images: list[bytes],
        prompt: str,
        session: aiohttp.ClientSession
    ) -> str:
        """Send multiple images in one request. Return extracted text."""
        pass

    @abstractmethod
    def supports_batch(self) -> bool:
        """Does this provider support multi-image requests?"""
        pass

    def estimate_cost(self, num_images: int, dpi: int) -> Optional[float]:
        """Estimate cost. Override in subclasses that know pricing. Return None if unknown."""
        return None
