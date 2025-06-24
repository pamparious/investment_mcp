"""Abstract base class for AI providers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class BaseAIProvider(ABC):
    """Abstract base class for all AI providers."""
    
    def __init__(self, model_name: str, temperature: float = 0.1, max_tokens: int = 2048):
        """
        Initialize the AI provider.
        
        Args:
            model_name: Name of the model to use
            temperature: Temperature for AI generation (0.0-1.0)
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def analyze_market_data(
        self, 
        symbol: str, 
        market_data: Dict[str, Any], 
        technical_indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze market data and provide insights.
        
        Args:
            symbol: Stock symbol
            market_data: Current market data
            technical_indicators: Technical analysis indicators
            
        Returns:
            Dictionary containing AI analysis results
        """
        pass
    
    @abstractmethod
    async def generate_insights(
        self, 
        data_context: Dict[str, Any], 
        analysis_type: str = "market"
    ) -> Dict[str, Any]:
        """
        Generate insights from provided data context.
        
        Args:
            data_context: Context data for analysis
            analysis_type: Type of analysis (market, economic, portfolio)
            
        Returns:
            Dictionary containing generated insights
        """
        pass
    
    @abstractmethod
    async def explain_patterns(
        self, 
        pattern_data: Dict[str, Any], 
        pattern_type: str
    ) -> str:
        """
        Explain detected patterns in human-readable format.
        
        Args:
            pattern_data: Data about detected patterns
            pattern_type: Type of pattern (trend, correlation, etc.)
            
        Returns:
            Human-readable explanation of the patterns
        """
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """
        Check if the AI provider is available.
        
        Returns:
            True if provider is available, False otherwise
        """
        pass
    
    def _create_system_prompt(self, context: str) -> str:
        """
        Create a system prompt for financial analysis.
        
        Args:
            context: Context for the analysis
            
        Returns:
            System prompt string
        """
        return f"""You are a professional financial analyst with expertise in Swedish and international markets. 
        Provide clear, factual analysis based on the data provided. 
        Always format responses as valid JSON when requested.
        Focus on: {context}
        
        Be conservative in recommendations and always mention risks.
        Use precise financial terminology and provide specific insights."""
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response from AI, with error handling.
        
        Args:
            response: Raw response string
            
        Returns:
            Parsed JSON dictionary or error structure
        """
        import json
        
        try:
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # If no JSON found, return the text as analysis
                return {
                    "analysis": response,
                    "structured": False,
                    "error": "No JSON structure found in response"
                }
                
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON response: {e}")
            return {
                "analysis": response,
                "structured": False,
                "error": f"JSON parsing error: {str(e)}"
            }
        except Exception as e:
            self.logger.error(f"Unexpected error parsing response: {e}")
            return {
                "analysis": "Error processing AI response",
                "structured": False,
                "error": f"Unexpected error: {str(e)}"
            }