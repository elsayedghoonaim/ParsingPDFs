import base64
import logging
import aiohttp
from typing import Optional
from .base import BaseProvider

logger = logging.getLogger("pdf_extraction")


class OllamaProvider(BaseProvider):
    """Ollama VLM provider for local models like Llava, Bakllava, etc."""

    def __init__(self, config):
        super().__init__(config)
        self.base_url = config.provider_base_url or "http://localhost:11434"
        self.model = config.provider_model

    async def send_image(
        self,
        image_bytes: bytes,
        prompt: str,
        session: aiohttp.ClientSession
    ) -> str:
        """Sends a single image to Ollama's local generation endpoint."""
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [b64_image],
            "stream": False
        }

        url = f"{self.base_url}/api/generate"

        try:
            async with session.post(url, json=payload) as response:
                status = response.status
                response_text = await response.text()
                if status != 200:
                    logger.error(f"Ollama API error {status}: {response_text}")
                    raise RuntimeError(f"Ollama API error {status}: {response_text}")
                
                result = await response.json()
                return result["response"]
        except Exception as e:
            logger.error(f"Failed to send image to Ollama: {e}")
            raise

    async def send_images_batch(
        self,
        images: list[bytes],
        prompt: str,
        session: aiohttp.ClientSession
    ) -> str:
        """Sequential fallback for Ollama because Ollama doesn't support multi-image batching natively."""
        logger.warning("Ollama does not support batch mode. Falling back to sequential calls.")
        results = []
        for i, img_bytes in enumerate(images):
            try:
                res = await self.send_image(img_bytes, prompt, session)
                results.append(res)
            except Exception as e:
                logger.error(f"Error in Ollama batch sequential call at page index {i}: {e}")
                results.append(f"[Ollama Error: Page extraction failed: {e}]")
        return "\n\n---\n\n".join(results)

    def supports_batch(self) -> bool:
        """Ollama does not support batching multiple images in a single call."""
        return False

    def estimate_cost(self, num_images: int, dpi: int) -> Optional[float]:
        """Local execution has no API costs."""
        return 0.0
