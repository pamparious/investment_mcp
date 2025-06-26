"""
Gemma 3:1B Local AI Provider for Investment MCP System.

This module provides optimized integration with Google's Gemma 3:1B model
through Ollama for local inference focused on financial analysis.
"""

import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from ..core.config import get_settings

logger = logging.getLogger(__name__)


class GemmaProvider:
    """Optimized Gemma 3:1B provider for local financial analysis."""
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.OLLAMA_BASE_URL
        self.model_name = "gemma2:1b"  # Specific Gemma 3:1B model
        self.timeout = 30  # Optimized for local inference
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Gemma-specific parameters optimized for financial analysis
        self.model_params = {
            "temperature": 0.3,  # Lower for more consistent financial advice
            "top_p": 0.8,        # Focused responses
            "top_k": 40,         # Balanced creativity vs consistency
            "repeat_penalty": 1.1,
            "num_ctx": 2048,     # Context window optimization
        }
    
    async def check_availability(self) -> bool:
        """Check if Gemma model is available through Ollama."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                # Check if Ollama is running
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status != 200:
                        return False
                    
                    models = await response.json()
                    available_models = [model.get("name", "") for model in models.get("models", [])]
                    
                    # Check if Gemma 3:1B is available
                    gemma_available = any("gemma2:1b" in model or "gemma2" in model for model in available_models)
                    
                    if not gemma_available:
                        self.logger.warning("Gemma 3:1B model not found in Ollama. Available models: %s", available_models)
                    
                    return gemma_available
                    
        except Exception as e:
            self.logger.error(f"Failed to check Gemma availability: {e}")
            return False
    
    async def generate_portfolio_analysis(
        self, 
        portfolio_data: Dict[str, Any],
        market_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate portfolio analysis using Gemma 3:1B."""
        
        prompt = self._create_portfolio_analysis_prompt(portfolio_data, market_context)
        
        try:
            response = await self._call_gemma(prompt)
            
            if response["success"]:
                parsed_analysis = self._parse_portfolio_analysis(response["content"])
                return {
                    "success": True,
                    "analysis": parsed_analysis,
                    "model": self.model_name,
                    "inference_time": response.get("inference_time", 0)
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error", "Unknown error"),
                    "fallback_available": True
                }
                
        except Exception as e:
            self.logger.error(f"Portfolio analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_available": True
            }
    
    async def generate_risk_explanation(
        self, 
        risk_metrics: Dict[str, float],
        portfolio_allocation: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate natural language risk explanation."""
        
        prompt = self._create_risk_explanation_prompt(risk_metrics, portfolio_allocation)
        
        try:
            response = await self._call_gemma(prompt)
            
            if response["success"]:
                explanation = self._parse_risk_explanation(response["content"])
                return {
                    "success": True,
                    "explanation": explanation,
                    "model": self.model_name
                }
            else:
                return self._fallback_risk_explanation(risk_metrics)
                
        except Exception as e:
            self.logger.error(f"Risk explanation failed: {e}")
            return self._fallback_risk_explanation(risk_metrics)
    
    async def generate_market_commentary(
        self, 
        swedish_economic_data: Dict[str, Any],
        fund_performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate market commentary focusing on Swedish context."""
        
        prompt = self._create_market_commentary_prompt(swedish_economic_data, fund_performance)
        
        try:
            response = await self._call_gemma(prompt)
            
            if response["success"]:
                commentary = self._parse_market_commentary(response["content"])
                return {
                    "success": True,
                    "commentary": commentary,
                    "model": self.model_name
                }
            else:
                return self._fallback_market_commentary(swedish_economic_data)
                
        except Exception as e:
            self.logger.error(f"Market commentary failed: {e}")
            return self._fallback_market_commentary(swedish_economic_data)
    
    async def _call_gemma(self, prompt: str) -> Dict[str, Any]:
        """Make optimized call to Gemma 3:1B model."""
        
        start_time = datetime.now()
        
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": self.model_params
                }
                
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        inference_time = (datetime.now() - start_time).total_seconds()
                        
                        return {
                            "success": True,
                            "content": result.get("response", ""),
                            "inference_time": inference_time,
                            "tokens": len(result.get("response", "").split())
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}: {error_text}"
                        }
                        
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Request timeout - Gemma inference took too long"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Gemma call failed: {str(e)}"
            }
    
    def _create_portfolio_analysis_prompt(
        self, 
        portfolio_data: Dict[str, Any],
        market_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create optimized prompt for portfolio analysis."""
        
        # Gemma-specific prompt structure - concise and focused
        prompt = f"""Du är en svensk investeringsrådgivare. Analysera denna portfölj:

PORTFÖLJ:
{self._format_portfolio_for_prompt(portfolio_data)}

UPPGIFT:
1. Bedöm risk (Hög/Medium/Låg)
2. Föreslå förbättringar 
3. Kommentera svenska marknaden
4. Ge konkreta råd

SVAR (max 200 ord):"""
        
        return prompt
    
    def _create_risk_explanation_prompt(
        self, 
        risk_metrics: Dict[str, float],
        portfolio_allocation: Dict[str, float]
    ) -> str:
        """Create prompt for risk explanation."""
        
        sharpe = risk_metrics.get("sharpe_ratio", 0)
        volatility = risk_metrics.get("annualized_volatility", 0) * 100
        max_dd = abs(risk_metrics.get("max_drawdown", 0)) * 100
        
        prompt = f"""Förklara portföljrisk på svenska för privatinvesterare:

RISKMÅTT:
- Sharpe-kvot: {sharpe:.2f}
- Volatilitet: {volatility:.1f}%
- Max nedgång: {max_dd:.1f}%

Förklara vad detta betyder för en svensk investerare i enkla termer (max 150 ord):"""
        
        return prompt
    
    def _create_market_commentary_prompt(
        self, 
        swedish_economic_data: Dict[str, Any],
        fund_performance: Dict[str, Any]
    ) -> str:
        """Create prompt for Swedish market commentary."""
        
        prompt = f"""Kommentera svensk marknad för investerare:

EKONOMISK DATA:
{self._format_economic_data_for_prompt(swedish_economic_data)}

FONDPRESTATION:
{self._format_fund_performance_for_prompt(fund_performance)}

Ge marknadskommentar på svenska (max 200 ord):
1. Läget just nu
2. Vad investerare bör tänka på
3. Konkreta råd"""
        
        return prompt
    
    def _format_portfolio_for_prompt(self, portfolio_data: Dict[str, Any]) -> str:
        """Format portfolio data for Gemma prompt."""
        
        allocation = portfolio_data.get("allocation", {})
        lines = []
        
        for fund_id, weight in allocation.items():
            fund_name = fund_id.replace("_", " ").title()
            lines.append(f"- {fund_name}: {weight*100:.1f}%")
        
        return "\n".join(lines[:5])  # Limit to top 5 for prompt efficiency
    
    def _format_economic_data_for_prompt(self, economic_data: Dict[str, Any]) -> str:
        """Format Swedish economic data for prompt."""
        
        lines = []
        for key, value in economic_data.items():
            if isinstance(value, (int, float)):
                lines.append(f"- {key}: {value}")
        
        return "\n".join(lines[:4])  # Limit for prompt efficiency
    
    def _format_fund_performance_for_prompt(self, fund_performance: Dict[str, Any]) -> str:
        """Format fund performance for prompt."""
        
        lines = []
        for fund, perf in fund_performance.items():
            if isinstance(perf, dict) and "return" in perf:
                lines.append(f"- {fund}: {perf['return']*100:.1f}%")
        
        return "\n".join(lines[:4])
    
    def _parse_portfolio_analysis(self, content: str) -> Dict[str, Any]:
        """Parse Gemma's portfolio analysis response."""
        
        # Simple parsing for structured output
        lines = content.strip().split('\n')
        
        analysis = {
            "risk_assessment": "Medium",  # Default
            "recommendations": [],
            "swedish_context": "",
            "overall_rating": "Neutral"
        }
        
        # Extract key information
        for line in lines:
            line = line.strip()
            if "risk" in line.lower() or "risk" in line.lower():
                if "hög" in line.lower() or "high" in line.lower():
                    analysis["risk_assessment"] = "High"
                elif "låg" in line.lower() or "low" in line.lower():
                    analysis["risk_assessment"] = "Low"
            
            if "föreslå" in line.lower() or "rekommendera" in line.lower():
                analysis["recommendations"].append(line)
        
        analysis["full_analysis"] = content
        return analysis
    
    def _parse_risk_explanation(self, content: str) -> Dict[str, Any]:
        """Parse risk explanation response."""
        
        return {
            "explanation": content.strip(),
            "risk_level": self._extract_risk_level(content),
            "key_points": self._extract_key_points(content)
        }
    
    def _parse_market_commentary(self, content: str) -> Dict[str, Any]:
        """Parse market commentary response."""
        
        return {
            "commentary": content.strip(),
            "market_outlook": self._extract_market_outlook(content),
            "recommendations": self._extract_recommendations(content)
        }
    
    def _extract_risk_level(self, content: str) -> str:
        """Extract risk level from Swedish text."""
        
        content_lower = content.lower()
        if "hög risk" in content_lower or "riskabel" in content_lower:
            return "Hög"
        elif "låg risk" in content_lower or "säker" in content_lower:
            return "Låg"
        else:
            return "Medium"
    
    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from explanation."""
        
        points = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                points.append(line[1:].strip())
        
        return points[:3]  # Limit to 3 key points
    
    def _extract_market_outlook(self, content: str) -> str:
        """Extract market outlook from commentary."""
        
        content_lower = content.lower()
        if "positiv" in content_lower or "bra" in content_lower:
            return "Positive"
        elif "negativ" in content_lower or "dålig" in content_lower:
            return "Negative"
        else:
            return "Neutral"
    
    def _extract_recommendations(self, content: str) -> List[str]:
        """Extract recommendations from commentary."""
        
        recommendations = []
        lines = content.split('\n')
        
        for line in lines:
            if "bör" in line.lower() or "ska" in line.lower() or "rekommendera" in line.lower():
                recommendations.append(line.strip())
        
        return recommendations[:3]
    
    def _fallback_risk_explanation(self, risk_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Provide fallback risk explanation when AI fails."""
        
        sharpe = risk_metrics.get("sharpe_ratio", 0)
        volatility = risk_metrics.get("annualized_volatility", 0) * 100
        
        if sharpe > 1.0:
            risk_level = "Låg till Medium"
            explanation = f"Portföljen har bra riskjusterad avkastning (Sharpe {sharpe:.2f}). Volatiliteten på {volatility:.1f}% är rimlig."
        elif sharpe > 0.5:
            risk_level = "Medium"
            explanation = f"Portföljen har acceptabel riskjusterad avkastning. Volatiliteten på {volatility:.1f}% kräver övervakning."
        else:
            risk_level = "Medium till Hög"
            explanation = f"Portföljen har låg riskjusterad avkastning (Sharpe {sharpe:.2f}). Volatiliteten på {volatility:.1f}% är hög."
        
        return {
            "success": True,
            "explanation": {
                "explanation": explanation,
                "risk_level": risk_level,
                "key_points": [
                    f"Sharpe-kvot: {sharpe:.2f}",
                    f"Årlig volatilitet: {volatility:.1f}%",
                    "Baserat på matematisk analys"
                ]
            },
            "fallback": True
        }
    
    def _fallback_market_commentary(self, economic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide fallback market commentary when AI fails."""
        
        commentary = "Marknadsläget analyseras baserat på tillgänglig ekonomisk data. " \
                    "Följ Riksbankens räntebesked och svenska ekonomiska indikatorer för investeringsbeslut."
        
        return {
            "success": True,
            "commentary": {
                "commentary": commentary,
                "market_outlook": "Neutral",
                "recommendations": [
                    "Diversifiera över olika tillgångsklasser",
                    "Överväg svensk ränteutveckling", 
                    "Följ bostadsmarknadens utveckling"
                ]
            },
            "fallback": True
        }
    
    async def pull_model_if_needed(self) -> bool:
        """Pull Gemma 3:1B model if not available."""
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
                payload = {"name": self.model_name}
                
                async with session.post(f"{self.base_url}/api/pull", json=payload) as response:
                    if response.status == 200:
                        self.logger.info("Successfully pulled Gemma 3:1B model")
                        return True
                    else:
                        self.logger.error(f"Failed to pull model: {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Error pulling Gemma model: {e}")
            return False