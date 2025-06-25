"""AI configuration and provider factory - Ollama only."""

import logging
from typing import Optional, Dict, Any
from .providers import BaseAIProvider, OllamaProvider

logger = logging.getLogger(__name__)


class AIConfig:
    """AI configuration and provider factory."""
    
    def __init__(self, settings: Optional[Any] = None):
        """
        Initialize AI configuration.
        
        Args:
            settings: Settings object containing AI configuration
        """
        self.settings = settings
        self._current_provider: Optional[BaseAIProvider] = None
    
    def get_provider(self, provider_name: Optional[str] = None) -> BaseAIProvider:
        """
        Get AI provider instance - only Ollama supported.
        
        Args:
            provider_name: Must be 'ollama' or None (ignored)
            
        Returns:
            Configured Ollama provider instance
        """
        # Only Ollama is supported now
        return self._create_ollama_provider()
    
    def _create_ollama_provider(self) -> OllamaProvider:
        """Create Ollama provider with configuration."""
        if self.settings:
            model_name = getattr(self.settings, 'OLLAMA_MODEL', 'gemma3:1b')
            base_url = getattr(self.settings, 'OLLAMA_BASE_URL', 'http://localhost:11434')
            temperature = getattr(self.settings, 'ANALYSIS_TEMPERATURE', 0.3)
            max_tokens = getattr(self.settings, 'ANALYSIS_MAX_TOKENS', 1024)
            timeout = getattr(self.settings, 'OLLAMA_TIMEOUT', 30)
        else:
            model_name = 'gemma3:1b'
            base_url = 'http://localhost:11434'
            temperature = 0.3
            max_tokens = 1024
            timeout = 30
        
        return OllamaProvider(
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
    
    
    async def test_provider(self) -> Dict[str, Any]:
        """
        Test if Ollama provider is available and working.
        
        Returns:
            Test results dictionary
        """
        try:
            provider = self.get_provider()
            
            # Test availability directly
            is_available = await provider.is_available()
            
            return {
                "provider": "ollama",
                "available": is_available,
                "model": provider.model_name,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error testing Ollama provider: {e}")
            return {
                "provider": "ollama",
                "available": False,
                "model": None,
                "error": str(e)
            }