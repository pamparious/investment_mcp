"""
AI package for Investment MCP System - Phase 4.

Local AI integration focused on Gemma 3:1B through Ollama for
Swedish investment analysis and portfolio optimization.
"""

from .gemma_provider import GemmaProvider
from .prompt_templates import GemmaPrompts
from .response_parser import GemmaResponseParser
from .ai_engine import AIEngine

__all__ = [
    'GemmaProvider',
    'GemmaPrompts', 
    'GemmaResponseParser',
    'AIEngine'
]