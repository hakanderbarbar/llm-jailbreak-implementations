"""
Thin wrapper around the OpenRouter APIs.
"""
import requests
import logging
import os
from typing import Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()
API_KEY=os.getenv("OPENROUTER_API_KEY")

class OpenRouterClient:
    def __init__(self):
        self.base_url = "https://openrouter.ai/api/v1"
        return
    
    def chat(
        self,
        model: str,
        messages: list,
        system: Optional[str] = None,
        temperature: float = 0.7,
        seed: int = 42,
        timeout: int = 120,
    ) -> str:
        """
        Send a chat request to an LLM through OpenRouter and return the assistant reply.

        Args:
            model:       Ollama model name, e.g. "llama3.1", "mistral"
            messages:    List of {"role": ..., "content": ...} dicts
            system:      Optional system prompt (prepended automatically)
            temperature: Sampling temperature
            seed:        RNG seed for reproducibility
            timeout:     Request timeout in seconds
        """
        if system:
            # Prepend system role if not already first message
            if not messages or messages[0]["role"] != "system":
                messages = [{"role": "system", "content": system}] + messages

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "seed": seed,
        }

        headers = {
            "Authorization": f"Bearer {API_KEY}",
        }

        resp = requests.post(
            f'{self.base_url}/chat/completions',
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        try:
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            logger.error(f"Timeout querying {model} after {timeout}s")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error querying {model}: {e}")
            logger.error("OpenRouter status: %s", resp.status_code)
            logger.error("OpenRouter response: %s", resp.text)
            logger.error("Payload sent: %s", payload)
            raise

        data = resp.json()
        return data["choices"][0]["message"]["content"]