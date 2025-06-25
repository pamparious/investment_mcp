"""Ollama AI provider implementation."""

import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, Optional
from .base_provider import BaseAIProvider

logger = logging.getLogger(__name__)


class OllamaProvider(BaseAIProvider):
    """Ollama local AI provider implementation."""
    
    def __init__(
        self, 
        model_name: str = "gemma3:1b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout: int = 30
    ):
        """
        Initialize Ollama provider.
        
        Args:
            model_name: Ollama model name
            base_url: Ollama server base URL
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        super().__init__(model_name, temperature, max_tokens)
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={"Content-Type": "application/json"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            if not self.session:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/api/tags") as response:
                        return response.status == 200
            else:
                async with self.session.get(f"{self.base_url}/api/tags") as response:
                    return response.status == 200
        except Exception as e:
            self.logger.warning(f"Ollama not available: {e}")
            return False
    
    async def _generate_completion(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generate completion using Ollama API.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            
        Returns:
            Generated text response
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")
        
        url = f"{self.base_url}/api/generate"
        
        # Combine system and user prompts
        full_prompt = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "").strip()
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
                    
        except asyncio.TimeoutError:
            raise Exception(f"Ollama request timed out after {self.timeout} seconds")
        except aiohttp.ClientError as e:
            raise Exception(f"Ollama connection error: {e}")
    
    async def analyze_market_data(
        self, 
        symbol: str, 
        market_data: Dict[str, Any], 
        technical_indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze market data using Ollama."""
        system_prompt = self._create_system_prompt("market analysis and stock evaluation")
        
        # Format the data for analysis
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
        - Support Level: ${technical_indicators.get('support', 'N/A')}
        - Resistance Level: ${technical_indicators.get('resistance', 'N/A')}

        Provide analysis in JSON format:
        {{
          "assessment": "bullish/bearish/neutral",
          "confidence": 0.8,
          "key_levels": {{"support": 123.45, "resistance": 134.56}},
          "risks": ["risk1", "risk2"],
          "opportunities": ["opportunity1", "opportunity2"],
          "outlook": "short-term outlook description",
          "recommendation": "buy/hold/sell",
          "reasoning": "detailed reasoning for recommendation"
        }}
        """
        
        try:
            response = await self._generate_completion(prompt, system_prompt)
            parsed_response = self._parse_json_response(response)
            
            # Add metadata
            parsed_response["provider"] = "ollama"
            parsed_response["model"] = self.model_name
            parsed_response["symbol"] = symbol
            
            return parsed_response
            
        except Exception as e:
            self.logger.error(f"Error analyzing market data for {symbol}: {e}")
            return {
                "error": str(e),
                "symbol": symbol,
                "provider": "ollama",
                "analysis": "Failed to generate analysis due to AI provider error"
            }
    
    async def generate_insights(
        self, 
        data_context: Dict[str, Any], 
        analysis_type: str = "market"
    ) -> Dict[str, Any]:
        """Generate insights from data context."""
        system_prompt = self._create_system_prompt(f"{analysis_type} insight generation")
        
        if analysis_type == "economic":
            prompt = f"""
            Analyze this Swedish economic data:
            
            {json.dumps(data_context, indent=2)}
            
            Provide economic insights in JSON format:
            {{
              "economic_outlook": "positive/negative/neutral",
              "key_indicators": ["indicator1", "indicator2"],
              "inflation_trend": "description",
              "currency_strength": "strong/moderate/weak",
              "housing_market": "analysis of housing trends",
              "investment_implications": ["implication1", "implication2"],
              "risks": ["economic risk1", "economic risk2"]
            }}
            """
        elif analysis_type == "portfolio":
            prompt = f"""
            Analyze this portfolio context:
            
            {json.dumps(data_context, indent=2)}
            
            Provide portfolio insights in JSON format:
            {{
              "diversification_score": 0.75,
              "risk_level": "low/medium/high",
              "recommended_allocation": {{"stocks": 60, "bonds": 30, "cash": 10}},
              "rebalancing_needed": true,
              "suggested_actions": ["action1", "action2"],
              "risk_mitigation": ["strategy1", "strategy2"]
            }}
            """
        elif analysis_type == "constrained_portfolio_recommendation":
            # Extract fund context for AI analysis
            fund_context = data_context.get('available_funds', {})
            investment_params = data_context.get('investment_parameters', {})
            constraints = data_context.get('constraints', {})
            base_suggestion = data_context.get('base_suggestion', {})
            
            approved_fund_list = constraints.get('approved_fund_ids', [])
            fund_details = fund_context.get('fund_details', [])
            
            # Create simplified prompt for faster processing
            risk = investment_params.get('risk_tolerance', 'balanced')
            amount = investment_params.get('amount', 0)
            
            # Get top 6 fund names for simplified context
            top_funds = [f['id'] for f in fund_details[:6]]
            
            prompt = f"""Create portfolio allocation for {amount:,} SEK, {risk} risk tolerance.
            
Use ONLY these funds: {top_funds[:6]}
Allocations must sum to 100%.

Respond with JSON only:
{{
  "recommended_allocation": {{"FUND_ID": 0.XX}},
  "reasoning": "Brief explanation"
}}"""
        else:  # market
            prompt = f"""
            Analyze this market context:
            
            {json.dumps(data_context, indent=2)}
            
            Provide market insights in JSON format:
            {{
              "market_sentiment": "bullish/bearish/neutral",
              "volatility_assessment": "low/medium/high",
              "sector_rotation": "description of sector trends",
              "global_factors": ["factor1", "factor2"],
              "trading_opportunities": ["opportunity1", "opportunity2"],
              "market_risks": ["risk1", "risk2"]
            }}
            """
        
        try:
            response = await self._generate_completion(prompt, system_prompt)
            parsed_response = self._parse_json_response(response)
            
            parsed_response["provider"] = "ollama"
            parsed_response["analysis_type"] = analysis_type
            
            return parsed_response
            
        except Exception as e:
            self.logger.error(f"Error generating {analysis_type} insights: {e}")
            return {
                "error": str(e),
                "analysis_type": analysis_type,
                "provider": "ollama",
                "insights": f"Failed to generate {analysis_type} insights"
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
        
        Pattern Data:
        {json.dumps(pattern_data, indent=2)}
        
        Provide a clear, educational explanation of what this pattern means for investors.
        Include:
        1. What the pattern indicates
        2. Historical significance
        3. Potential implications
        4. What investors should watch for
        
        Keep the explanation accessible but professional.
        """
        
        try:
            response = await self._generate_completion(prompt, system_prompt)
            return response
            
        except Exception as e:
            self.logger.error(f"Error explaining {pattern_type} pattern: {e}")
            return f"Unable to explain {pattern_type} pattern due to AI provider error: {str(e)}"