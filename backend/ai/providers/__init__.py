"""AI providers module - Ollama only."""

from .base_provider import BaseAIProvider
from .ollama_provider import OllamaProvider

__all__ = [
    'BaseAIProvider',
    'OllamaProvider'
]