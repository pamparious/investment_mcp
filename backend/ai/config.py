"""AI configuration and provider factory."""

import logging
from typing import Optional, Dict, Any
from .providers import BaseAIProvider, OllamaProvider

logger = logging.getLogger(__name__)

# Import optional providers safely
try:
    from .providers import OpenAIProvider
except ImportError:
    OpenAIProvider = None

try:
    from .providers import ClaudeProvider
except ImportError:
    ClaudeProvider = None


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
        Get AI provider instance based on configuration.
        
        Args:
            provider_name: Override provider name
            
        Returns:
            Configured AI provider instance
            
        Raises:
            ValueError: If provider is not available or configured
        """
        if not provider_name and self.settings:
            provider_name = getattr(self.settings, 'AI_PROVIDER', 'ollama')
        elif not provider_name:
            provider_name = 'ollama'  # Default fallback
        
        provider_name = provider_name.lower()
        
        if provider_name == 'ollama':
            return self._create_ollama_provider()
        elif provider_name == 'openai':
            return self._create_openai_provider()
        elif provider_name == 'claude':
            return self._create_claude_provider()
        else:
            raise ValueError(f"Unknown AI provider: {provider_name}")
    
    def _create_ollama_provider(self) -> OllamaProvider:
        """Create Ollama provider with configuration."""
        if self.settings:
            model_name = getattr(self.settings, 'OLLAMA_MODEL', 'gemma3:1b')
            base_url = getattr(self.settings, 'OLLAMA_BASE_URL', 'http://localhost:11434')
            temperature = getattr(self.settings, 'ANALYSIS_TEMPERATURE', 0.1)
            max_tokens = getattr(self.settings, 'ANALYSIS_MAX_TOKENS', 2048)
        else:
            model_name = 'gemma3:1b'
            base_url = 'http://localhost:11434'
            temperature = 0.1
            max_tokens = 2048
        
        return OllamaProvider(
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def _create_openai_provider(self) -> BaseAIProvider:
        """Create OpenAI provider with configuration."""
        if not OpenAIProvider:
            raise ValueError("OpenAI provider not available. Install with: pip install openai")
        
        if self.settings:
            api_key = getattr(self.settings, 'OPENAI_API_KEY', None)
            model_name = getattr(self.settings, 'OPENAI_MODEL', 'gpt-3.5-turbo')
            temperature = getattr(self.settings, 'ANALYSIS_TEMPERATURE', 0.1)
            max_tokens = getattr(self.settings, 'ANALYSIS_MAX_TOKENS', 2048)
        else:
            api_key = None
            model_name = 'gpt-3.5-turbo'
            temperature = 0.1
            max_tokens = 2048
        
        if not api_key:
            raise ValueError("OpenAI API key not configured")
        
        return OpenAIProvider(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def _create_claude_provider(self) -> BaseAIProvider:
        """Create Claude provider with configuration."""
        if not ClaudeProvider:
            raise ValueError("Claude provider not available. Install with: pip install anthropic")
        
        if self.settings:
            api_key = getattr(self.settings, 'ANTHROPIC_API_KEY', None)
            model_name = getattr(self.settings, 'CLAUDE_MODEL', 'claude-3-haiku-20240307')
            temperature = getattr(self.settings, 'ANALYSIS_TEMPERATURE', 0.1)
            max_tokens = getattr(self.settings, 'ANALYSIS_MAX_TOKENS', 2048)
        else:
            api_key = None
            model_name = 'claude-3-haiku-20240307'
            temperature = 0.1
            max_tokens = 2048
        
        if not api_key:
            raise ValueError("Anthropic API key not configured")
        
        return ClaudeProvider(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    async def test_provider(self, provider_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Test if a provider is available and working.
        
        Args:
            provider_name: Provider to test
            
        Returns:
            Test results dictionary
        """
        try:
            provider = self.get_provider(provider_name)
            
            # Test availability
            if hasattr(provider, '__aenter__'):
                async with provider:
                    is_available = await provider.is_available()
            else:
                is_available = await provider.is_available()
            
            return {
                "provider": provider_name or getattr(self.settings, 'AI_PROVIDER', 'ollama'),
                "available": is_available,
                "model": provider.model_name,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error testing provider {provider_name}: {e}")
            return {
                "provider": provider_name or "unknown",
                "available": False,
                "model": None,
                "error": str(e)
            }
    
    def list_available_providers(self) -> Dict[str, bool]:
        """
        List all available providers.
        
        Returns:
            Dictionary of provider names and availability
        """
        providers = {
            "ollama": True,  # Always available as it's the default
            "openai": OpenAIProvider is not None,
            "claude": ClaudeProvider is not None
        }
        
        return providers