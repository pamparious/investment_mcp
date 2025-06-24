"""Anthropic Claude provider implementation."""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from .base_provider import BaseAIProvider

logger = logging.getLogger(__name__)

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic library not available. Install with: pip install anthropic")


class ClaudeProvider(BaseAIProvider):
    """Anthropic Claude API provider implementation."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "claude-3-haiku-20240307",
        temperature: float = 0.1,
        max_tokens: int = 2048
    ):
        """
        Initialize Claude provider.
        
        Args:
            api_key: Anthropic API key
            model_name: Claude model name
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__(model_name, temperature, max_tokens)
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")
        
        if not api_key:
            raise ValueError("Anthropic API key is required")
        
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    async def is_available(self) -> bool:
        """Check if Claude API is available."""
        if not ANTHROPIC_AVAILABLE:
            return False
        
        try:
            # Test with a simple message
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=1,
                messages=[{"role": "user", "content": "test"}]
            )
            return bool(response.content)
        except Exception as e:
            self.logger.warning(f"Claude API not available: {e}")
            return False
    
    async def _generate_completion(self, prompt: str, system_prompt: str = "") -> str:
        """Generate completion using Claude API."""
        try:
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt if system_prompt else None,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract text from response
            if response.content and len(response.content) > 0:
                return response.content[0].text.strip()
            else:
                return ""
            
        except Exception as e:
            raise Exception(f"Claude API error: {e}")
    
    async def analyze_market_data(
        self, 
        symbol: str, 
        market_data: Dict[str, Any], 
        technical_indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze market data using Claude."""
        system_prompt = self._create_system_prompt("market analysis and stock evaluation")
        
        current_price = market_data.get('close_price', market_data.get('current_price', 'N/A'))
        volume = market_data.get('volume', 'N/A')
        
        prompt = f"""
        Analyze this stock data for {symbol}:

        Current Price: ${current_price}
        Volume: {volume}
        
        Technical Indicators:
        - 20-day SMA: ${technical_indicators.get('sma_20', 'N/A')}
        - 50-day SMA: ${technical_indicators.get('sma_50', 'N/A')}
        - RSI: {technical_indicators.get('rsi', 'N/A')}
        - Trend Direction: {technical_indicators.get('trend', 'N/A')}

        Provide analysis in JSON format:
        {{
          "assessment": "bullish/bearish/neutral",
          "confidence": 0.8,
          "key_levels": {{"support": 123.45, "resistance": 134.56}},
          "risks": ["risk1", "risk2"],
          "opportunities": ["opportunity1", "opportunity2"],
          "outlook": "short-term outlook description",
          "recommendation": "buy/hold/sell",
          "reasoning": "detailed reasoning"
        }}
        """
        
        try:
            response = await self._generate_completion(prompt, system_prompt)
            parsed_response = self._parse_json_response(response)
            
            parsed_response["provider"] = "claude"
            parsed_response["model"] = self.model_name
            parsed_response["symbol"] = symbol
            
            return parsed_response
            
        except Exception as e:
            self.logger.error(f"Error analyzing market data for {symbol}: {e}")
            return {
                "error": str(e),
                "symbol": symbol,
                "provider": "claude",
                "analysis": "Failed to generate analysis"
            }
    
    async def generate_insights(
        self, 
        data_context: Dict[str, Any], 
        analysis_type: str = "market"
    ) -> Dict[str, Any]:
        """Generate insights from data context."""
        system_prompt = self._create_system_prompt(f"{analysis_type} insight generation")
        
        prompt = f"""
        Analyze this {analysis_type} data and provide insights:
        
        {json.dumps(data_context, indent=2)}
        
        Provide insights in structured JSON format appropriate for {analysis_type} analysis.
        """
        
        try:
            response = await self._generate_completion(prompt, system_prompt)
            parsed_response = self._parse_json_response(response)
            
            parsed_response["provider"] = "claude"
            parsed_response["analysis_type"] = analysis_type
            
            return parsed_response
            
        except Exception as e:
            self.logger.error(f"Error generating {analysis_type} insights: {e}")
            return {
                "error": str(e),
                "analysis_type": analysis_type,
                "provider": "claude"
            }
    
    async def explain_patterns(
        self, 
        pattern_data: Dict[str, Any], 
        pattern_type: str
    ) -> str:
        """Explain detected patterns."""
        system_prompt = self._create_system_prompt("pattern explanation and interpretation")
        
        prompt = f"""
        Explain this {pattern_type} pattern in the financial data:
        
        {json.dumps(pattern_data, indent=2)}
        
        Provide a clear explanation of what this pattern means for investors.
        """
        
        try:
            response = await self._generate_completion(prompt, system_prompt)
            return response
            
        except Exception as e:
            self.logger.error(f"Error explaining {pattern_type} pattern: {e}")
            return f"Unable to explain pattern due to API error: {str(e)}"