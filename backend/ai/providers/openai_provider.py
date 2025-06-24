"""OpenAI provider implementation."""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from .base_provider import BaseAIProvider

logger = logging.getLogger(__name__)

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. Install with: pip install openai")


class OpenAIProvider(BaseAIProvider):
    """OpenAI API provider implementation."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.1,
        max_tokens: int = 2048
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model_name: OpenAI model name
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__(model_name, temperature, max_tokens)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    async def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        if not OPENAI_AVAILABLE:
            return False
        
        try:
            # Test with a simple completion
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return bool(response.choices)
        except Exception as e:
            self.logger.warning(f"OpenAI API not available: {e}")
            return False
    
    async def _generate_completion(self, prompt: str, system_prompt: str = "") -> str:
        """Generate completion using OpenAI API."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
    
    async def analyze_market_data(
        self, 
        symbol: str, 
        market_data: Dict[str, Any], 
        technical_indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze market data using OpenAI."""
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
            
            parsed_response["provider"] = "openai"
            parsed_response["model"] = self.model_name
            parsed_response["symbol"] = symbol
            
            return parsed_response
            
        except Exception as e:
            self.logger.error(f"Error analyzing market data for {symbol}: {e}")
            return {
                "error": str(e),
                "symbol": symbol,
                "provider": "openai",
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
            
            parsed_response["provider"] = "openai"
            parsed_response["analysis_type"] = analysis_type
            
            return parsed_response
            
        except Exception as e:
            self.logger.error(f"Error generating {analysis_type} insights: {e}")
            return {
                "error": str(e),
                "analysis_type": analysis_type,
                "provider": "openai"
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