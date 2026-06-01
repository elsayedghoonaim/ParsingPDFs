import base64
import logging
import aiohttp
import os
from typing import Optional
from .base import BaseProvider

logger = logging.getLogger("pdf_extraction")


class VLLMProvider(BaseProvider):
    """vLLM provider using OpenAI-compatible chat completion API."""

    def __init__(self, config):
        super().__init__(config)
        self.base_url = config.provider_base_url or "http://localhost:8000/v1"
        self.api_key = config.provider_api_key or os.environ.get("VLLM_API_KEY")
        self.model = config.provider_model

    async def send_image(
        self,
        image_bytes: bytes,
        prompt: str,
        session: aiohttp.ClientSession
    ) -> str:
        """Sends a single image to vLLM's OpenAI-compatible completions endpoint."""
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
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
            "max_tokens": 4096
        }

        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"

        try:
            async with session.post(url, headers=headers, json=payload) as response:
                status = response.status
                response_text = await response.text()
                if status != 200:
                    logger.error(f"vLLM API error {status}: {response_text}")
                    raise RuntimeError(f"vLLM API error {status}: {response_text}")
                
                result = await response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to send image to vLLM: {e}")
            raise

    async def send_images_batch(
        self,
        images: list[bytes],
        prompt: str,
        session: aiohttp.ClientSession
    ) -> str:
        """Sends a batch of images to vLLM's OpenAI-compatible completions endpoint in a single request."""
        content_payload = [{"type": "text", "text": prompt}]
        for image_bytes in images:
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            content_payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_image}"}
            })

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content_payload
                }
            ],
            "max_tokens": 4096
        }

        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"

        try:
            async with session.post(url, headers=headers, json=payload) as response:
                status = response.status
                response_text = await response.text()
                if status != 200:
                    logger.error(f"vLLM API batch error {status}: {response_text}")
                    raise RuntimeError(f"vLLM API batch error {status}: {response_text}")
                
                result = await response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to send image batch to vLLM: {e}")
            raise

    def supports_batch(self) -> bool:
        """vLLM typically supports batching multiple images in a single call."""
        return True

    def estimate_cost(self, num_images: int, dpi: int) -> Optional[float]:
        """Local self-hosted execution has no API costs."""
        return 0.0
