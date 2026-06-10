"""
Thin wrapper around the Ollama REST API.
Docs: https://github.com/ollama/ollama/blob/main/docs/api.md
"""
import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

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
        Send a chat request to Ollama and return the assistant reply.

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
            "options": {
                "temperature": temperature,
                "seed": seed,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except requests.exceptions.Timeout:
            logger.error(f"Timeout querying {model} after {timeout}s")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error querying {model}: {e}")
            raise

    def list_models(self) -> list[str]:
        """Return list of available Ollama model names."""
        resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def check_models(self, *models: str) -> None:
        """Raise RuntimeError if any required model is not available."""
        available = self.list_models()
        # Normalise: "llama3.1" matches "llama3.1:latest"
        available_short = {m.split(":")[0] for m in available}
        for m in models:
            if m not in available and m.split(":")[0] not in available_short:
                raise RuntimeError(
                    f"Model '{m}' not found in Ollama. "
                    f"Run: ollama pull {m}\nAvailable: {available}"
                )
