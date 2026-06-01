import base64
import logging
import aiohttp
import os
from typing import Optional
from .base import BaseProvider

logger = logging.getLogger("pdf_extraction")


class OpenAIProvider(BaseProvider):
    """OpenAI VLM provider supporting GPT-4o, GPT-4o-mini, etc."""

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

    async def send_image(
        self,
        image_bytes: bytes,
        prompt: str,
        session: aiohttp.ClientSession
    ) -> str:
        """Sends a single image to OpenAI's Chat Completions endpoint."""
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
                    logger.error(f"OpenAI API error {status}: {response_text}")
                    raise RuntimeError(f"OpenAI API error {status}: {response_text}")
                
                result = await response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to send image to OpenAI: {e}")
            raise

    async def send_images_batch(
        self,
        images: list[bytes],
        prompt: str,
        session: aiohttp.ClientSession
    ) -> str:
        """Sends a batch of images to OpenAI's Chat Completions endpoint in a single request."""
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
                    logger.error(f"OpenAI API batch error {status}: {response_text}")
                    raise RuntimeError(f"OpenAI API batch error {status}: {response_text}")
                
                result = await response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to send image batch to OpenAI: {e}")
            raise

    def supports_batch(self) -> bool:
        """OpenAI models naturally support batching multiple images in a single call."""
        return True

    def estimate_cost(self, num_images: int, dpi: int) -> Optional[float]:
        """Estimates cost based on model and number of images/pages."""
        if self.config.cost_price_per_image is not None:
            return num_images * self.config.cost_price_per_image

        # Estimate based on models
        model_lower = self.model.lower()
        if "gpt-4o-mini" in model_lower:
            price = 0.001  # $0.001 per image
        elif "gpt-4o" in model_lower:
            price = 0.003  # $0.003 per image
        else:
            price = 0.003  # Fallback standard rate
        return num_images * price
