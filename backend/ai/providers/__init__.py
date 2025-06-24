"""AI providers module."""

from .base_provider import BaseAIProvider
from .ollama_provider import OllamaProvider

# Optional providers (only import if libraries are available)
try:
    from .openai_provider import OpenAIProvider
except ImportError:
    OpenAIProvider = None

try:
    from .claude_provider import ClaudeProvider
except ImportError:
    ClaudeProvider = None

__all__ = [
    'BaseAIProvider',
    'OllamaProvider'
]

# Add optional providers to exports if available
if OpenAIProvider:
    __all__.append('OpenAIProvider')
if ClaudeProvider:
    __all__.append('ClaudeProvider')