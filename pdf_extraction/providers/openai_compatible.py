"""Shared base class for OpenAI-compatible VLM providers (OpenAI, vLLM, Custom)."""

import base64
import logging
import aiohttp
from typing import Optional
from .base import BaseProvider, RateLimitError

logger = logging.getLogger("pdf_extraction")


class OpenAICompatibleProvider(BaseProvider):
    """
    Base implementation for any provider that speaks the OpenAI Chat Completions API format.

    Subclasses only need to:
      - Set self.base_url, self.api_key, self.model in __init__
      - Override supports_batch() and estimate_cost() as needed
      - Optionally set self.extra_headers for additional HTTP headers
      - Optionally set self._batch_supported to control batch support
    """

    # Subclasses set this to control supports_batch()
    _batch_supported: bool = True

    def __init__(self, config):
        super().__init__(config)
        # Subclasses must set: self.base_url, self.api_key, self.model
        self.extra_headers: dict = {}

    def _build_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def _build_single_payload(self, image_bytes: bytes, prompt: str) -> dict:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"}
                        }
                    ]
                }
            ],
            "max_tokens": self.config.vlm_max_output_tokens
        }

    def _build_batch_payload(self, images: list[bytes], prompt: str) -> dict:
        content_payload = [{"type": "text", "text": prompt}]
        for image_bytes in images:
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            content_payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_image}"}
            })
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content_payload}],
            "max_tokens": self.config.vlm_max_output_tokens
        }

    @staticmethod
    def _parse_response(result: dict) -> str:
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Invalid OpenAI-compatible API response format: {e}. Response: {result}")

    async def send_image(
        self,
        image_bytes: bytes,
        prompt: str,
        session: aiohttp.ClientSession
    ) -> str:
        """Sends a single image to the OpenAI-compatible completions endpoint."""
        url = f"{self.base_url}/chat/completions"
        payload = self._build_single_payload(image_bytes, prompt)
        headers = self._build_headers()

        async def _request():
            async with session.post(url, headers=headers, json=payload) as response:
                status = response.status
                if status != 200:
                    response_text = await response.text()
                    logger.debug(f"{self.__class__.__name__} API error body: {response_text}")
                    short = response_text[:200].split('\n')[0]
                    if status == 429:
                        logger.warning(f"{self.__class__.__name__} rate limited (HTTP 429)")
                        raise RateLimitError(f"{self.__class__.__name__} rate limited", retry_after=0.0)
                    logger.error(f"{self.__class__.__name__} API error {status}: {short}")
                    raise RuntimeError(f"{self.__class__.__name__} API error {status}: {short}")
                result = await response.json()
                return self._parse_response(result)

        return await self._retry_request(_request)

    async def send_images_batch(
        self,
        images: list[bytes],
        prompt: str,
        session: aiohttp.ClientSession
    ) -> str:
        """Sends a batch of images to the OpenAI-compatible completions endpoint."""
        url = f"{self.base_url}/chat/completions"
        payload = self._build_batch_payload(images, prompt)
        headers = self._build_headers()

        async def _request():
            async with session.post(url, headers=headers, json=payload) as response:
                status = response.status
                if status != 200:
                    response_text = await response.text()
                    logger.debug(f"{self.__class__.__name__} API batch error body: {response_text}")
                    short = response_text[:200].split('\n')[0]
                    if status == 429:
                        logger.warning(f"{self.__class__.__name__} rate limited (HTTP 429) — batch")
                        raise RateLimitError(f"{self.__class__.__name__} rate limited", retry_after=0.0)
                    logger.error(f"{self.__class__.__name__} API batch error {status}: {short}")
                    raise RuntimeError(f"{self.__class__.__name__} API batch error {status}: {short}")
                result = await response.json()
                return self._parse_response(result)

        return await self._retry_request(_request)

    def supports_batch(self) -> bool:
        return self._batch_supported

    def estimate_cost(self, num_images: int, dpi: int) -> Optional[float]:
        """Override in subclasses with provider-specific pricing."""
        return None
