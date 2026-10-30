"""Common message preparation for chat API adapters."""
from typing import Any


def prepare_messages(messages: list[dict[str, Any]], system: str | None = None) -> list[dict[str, Any]]:
    copied = [dict(message) for message in messages]
    if system and (not copied or copied[0].get('role') != 'system'):
        copied.insert(0, {'role': 'system', 'content': system})
    return copied
