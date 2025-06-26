"""
Main AI orchestrator for Investment MCP System Phase 4.

This module coordinates AI-powered analysis using Gemma 3:1B for
portfolio optimization, risk analysis, and investment insights.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .gemma_provider import GemmaProvider
from .prompt_templates import GemmaPrompts
from .response_parser import GemmaResponseParser
from ..core.config import get_settings, TRADEABLE_FUNDS

logger = logging.getLogger(__name__)


class AIEngine:
    """Main AI orchestrator for investment analysis."""
    
    def __init__(self):
        self.gemma = GemmaProvider()
        self.prompts = GemmaPrompts()
        self.parser = GemmaResponseParser()
        self.settings = get_settings()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance tracking
        self._analysis_cache = {}
        self._cache_ttl = 300  # 5 minutes
    
    async def initialize(self) -> bool:
        """Initialize AI engine and check Gemma availability."""
        
        try:
            # Check if Gemma is available
            available = await self.gemma.check_availability()
            
            if not available:
                self.logger.warning("Gemma 3:1B not available. Attempting to pull model...")
                pulled = await self.gemma.pull_model_if_needed()
                
                if pulled:
                    available = await self.gemma.check_availability()
            
            if available:
                self.logger.info("AI Engine initialized successfully with Gemma 3:1B")
                return True
            else:
                self.logger.error("Failed to initialize Gemma 3:1B model")
                return False
                
        except Exception as e:
            self.logger.error(f"AI Engine initialization failed: {e}")
            return False
    
    async def generate_portfolio_recommendation(
        self,
        current_allocation: Dict[str, float],
        risk_tolerance: str,
        investment_amount: float,
        market_context: Optional[Dict[str, Any]] = None,
        economic_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate AI-powered portfolio recommendation."""
        
        try:
            self.logger.info(f"Generating portfolio recommendation for {risk_tolerance} risk tolerance")
            
            # Create portfolio analysis context
            portfolio_data = {
                "allocation": current_allocation,
                "risk_tolerance": risk_tolerance,
                "investment_amount": investment_amount
            }
            
            # Generate AI analysis
            ai_result = await self.gemma.generate_portfolio_analysis(
                portfolio_data, market_context
            )
            
            if ai_result["success"]:
                # Parse the response
                parsed = self.parser.parse_portfolio_recommendation(ai_result["analysis"]["full_analysis"])
                
                # Enhance with quantitative validation
                enhanced_recommendation = await self._enhance_with_quantitative_analysis(
                    parsed, current_allocation, risk_tolerance
                )
                
                return {
                    "success": True,
                    "recommendation": enhanced_recommendation,
                    "ai_insights": ai_result["analysis"],
                    "model_used": "gemma2:1b",
                    "inference_time": ai_result.get("inference_time", 0),
                    "generated_at": datetime.now().isoformat()
                }
            else:
                # Fallback to quantitative-only recommendation
                return await self._fallback_recommendation(
                    current_allocation, risk_tolerance, investment_amount
                )
                
        except Exception as e:
            self.logger.error(f"Portfolio recommendation generation failed: {e}")
            return await self._fallback_recommendation(
                current_allocation, risk_tolerance, investment_amount
            )
    
    async def analyze_portfolio_risk(
        self,
        portfolio: Dict[str, float],
        risk_metrics: Dict[str, float],
        historical_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate AI-powered risk analysis."""
        
        try:
            self.logger.info("Generating AI risk analysis")
            
            # Generate risk explanation
            risk_result = await self.gemma.generate_risk_explanation(
                risk_metrics, portfolio
            )
            
            if risk_result["success"]:
                # Parse the response
                parsed = self.parser.parse_risk_analysis(risk_result["explanation"]["explanation"])
                
                return {
                    "success": True,
                    "risk_analysis": parsed,
                    "quantitative_metrics": risk_metrics,
                    "model_used": "gemma2:1b",
                    "generated_at": datetime.now().isoformat()
                }
            else:
                # Use fallback from Gemma provider
                return risk_result
                
        except Exception as e:
            self.logger.error(f"Risk analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_analysis": self._basic_risk_interpretation(risk_metrics)
            }
    
    async def generate_market_commentary(
        self,
        swedish_economic_data: Dict[str, Any],
        fund_performance: Dict[str, Any],
        market_indicators: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate AI-powered market commentary."""
        
        try:
            self.logger.info("Generating market commentary")
            
            # Generate commentary
            commentary_result = await self.gemma.generate_market_commentary(
                swedish_economic_data, fund_performance
            )
            
            if commentary_result["success"]:
                # Parse the response
                parsed = self.parser.parse_market_commentary(
                    commentary_result["commentary"]["commentary"]
                )
                
                return {
                    "success": True,
                    "market_commentary": parsed,
                    "economic_context": swedish_economic_data,
                    "model_used": "gemma2:1b",
                    "generated_at": datetime.now().isoformat()
                }
            else:
                # Use fallback
                return commentary_result
                
        except Exception as e:
            self.logger.error(f"Market commentary generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_commentary": self._basic_market_commentary(fund_performance)
            }
    
    async def generate_rebalancing_advice(
        self,
        current_portfolio: Dict[str, float],
        target_portfolio: Dict[str, float],
        market_conditions: str = "normal"
    ) -> Dict[str, Any]:
        """Generate AI-powered rebalancing advice."""
        
        try:
            # Create rebalancing prompt
            prompt = self.prompts.rebalancing_prompt(
                current_portfolio, target_portfolio, market_conditions
            )
            
            # Get AI response
            response = await self.gemma._call_gemma(prompt)
            
            if response["success"]:
                # Parse rebalancing advice
                parsed = self.parser.parse_rebalancing_advice(response["content"])
                
                return {
                    "success": True,
                    "rebalancing_advice": parsed,
                    "current_portfolio": current_portfolio,
                    "target_portfolio": target_portfolio,
                    "model_used": "gemma2:1b",
                    "generated_at": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error"),
                    "fallback_advice": self._basic_rebalancing_advice(
                        current_portfolio, target_portfolio
                    )
                }
                
        except Exception as e:
            self.logger.error(f"Rebalancing advice generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def analyze_housing_vs_investment(
        self,
        housing_data: Dict[str, Any],
        investment_returns: Dict[str, float],
        personal_situation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate housing vs investment analysis."""
        
        try:
            # Create housing analysis prompt
            prompt = self.prompts.housing_vs_investment_prompt(
                housing_data, investment_returns, personal_situation
            )
            
            # Get AI response
            response = await self.gemma._call_gemma(prompt)
            
            if response["success"]:
                return {
                    "success": True,
                    "housing_analysis": response["content"],
                    "recommendation_type": self._extract_housing_recommendation(response["content"]),
                    "model_used": "gemma2:1b",
                    "generated_at": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error"),
                    "fallback_analysis": self._basic_housing_analysis(
                        housing_data, investment_returns, personal_situation
                    )
                }
                
        except Exception as e:
            self.logger.error(f"Housing analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate_tax_optimization_advice(
        self,
        portfolio: Dict[str, float],
        realized_gains: float,
        investment_horizon: str
    ) -> Dict[str, Any]:
        """Generate Swedish tax optimization advice."""
        
        try:
            # Create tax optimization prompt
            prompt = self.prompts.tax_optimization_prompt(
                portfolio, realized_gains, investment_horizon
            )
            
            # Get AI response
            response = await self.gemma._call_gemma(prompt)
            
            if response["success"]:
                return {
                    "success": True,
                    "tax_advice": response["content"],
                    "key_strategies": self._extract_tax_strategies(response["content"]),
                    "model_used": "gemma2:1b",
                    "generated_at": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error"),
                    "fallback_advice": self._basic_tax_advice(portfolio, realized_gains)
                }
                
        except Exception as e:
            self.logger.error(f"Tax optimization advice failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _enhance_with_quantitative_analysis(
        self,
        ai_recommendation: Dict[str, Any],
        current_allocation: Dict[str, float],
        risk_tolerance: str
    ) -> Dict[str, Any]:
        """Enhance AI recommendation with quantitative validation."""
        
        # Validate AI allocation against fund universe
        ai_allocation = ai_recommendation.get("allocation", {})
        
        # Ensure all funds are in approved universe
        validated_allocation = {}
        for fund_id, weight in ai_allocation.items():
            if fund_id in TRADEABLE_FUNDS:
                validated_allocation[fund_id] = weight
        
        # Normalize allocation if needed
        total_weight = sum(validated_allocation.values())
        if total_weight > 0:
            validated_allocation = {
                k: v / total_weight for k, v in validated_allocation.items()
            }
        
        # If AI allocation is insufficient, use model portfolios as fallback
        if len(validated_allocation) < 3 or total_weight < 0.8:
            from ..analysis.portfolio import PortfolioOptimizer
            optimizer = PortfolioOptimizer()
            model_portfolios = optimizer.create_model_portfolios()
            
            # Select appropriate model portfolio
            portfolio_map = {
                "low": "conservative",
                "medium": "balanced",
                "high": "growth",
                "very_high": "aggressive"
            }
            
            model_type = portfolio_map.get(risk_tolerance, "balanced")
            if model_type in model_portfolios:
                validated_allocation = model_portfolios[model_type]["fund_allocation"]
        
        # Enhance recommendation
        ai_recommendation["allocation"] = validated_allocation
        ai_recommendation["quantitative_validation"] = True
        ai_recommendation["allocation_source"] = "ai_enhanced" if ai_allocation else "model_portfolio"
        
        return ai_recommendation
    
    async def _fallback_recommendation(
        self,
        current_allocation: Dict[str, float],
        risk_tolerance: str,
        investment_amount: float
    ) -> Dict[str, Any]:
        """Provide fallback recommendation when AI fails."""
        
        from ..analysis.portfolio import PortfolioOptimizer
        
        optimizer = PortfolioOptimizer()
        model_portfolios = optimizer.create_model_portfolios()
        
        # Select appropriate model portfolio
        portfolio_map = {
            "low": "conservative",
            "medium": "balanced", 
            "high": "growth",
            "very_high": "aggressive"
        }
        
        model_type = portfolio_map.get(risk_tolerance, "balanced")
        
        if model_type in model_portfolios:
            portfolio = model_portfolios[model_type]
            
            return {
                "success": True,
                "recommendation": {
                    "allocation": portfolio["fund_allocation"],
                    "risk_level": risk_tolerance,
                    "reasoning": [f"Modellportfölj för {risk_tolerance} risktolerans"],
                    "recommendations": ["Diversifierad allokering baserad på risktolerans"],
                    "confidence": 0.7,
                    "allocation_source": "model_portfolio"
                },
                "ai_insights": {"fallback": True, "reason": "AI unavailable"},
                "model_used": "quantitative_fallback",
                "generated_at": datetime.now().isoformat()
            }
        else:
            # Ultimate fallback
            return {
                "success": False,
                "error": "No suitable portfolio found",
                "fallback_available": False
            }
    
    def _basic_risk_interpretation(self, risk_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Basic risk interpretation fallback."""
        
        sharpe = risk_metrics.get("sharpe_ratio", 0)
        volatility = risk_metrics.get("annualized_volatility", 0) * 100
        
        if sharpe > 1.0:
            risk_level = "Låg till Medium"
        elif sharpe > 0.5:
            risk_level = "Medium"
        else:
            risk_level = "Medium till Hög"
        
        return {
            "risk_level": risk_level,
            "interpretation": f"Portföljen har Sharpe-kvot på {sharpe:.2f} och volatilitet på {volatility:.1f}%",
            "source": "quantitative_analysis"
        }
    
    def _basic_market_commentary(self, fund_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Basic market commentary fallback."""
        
        return {
            "commentary": "Marknadsanalys baserad på fondprestation och ekonomiska indikatorer.",
            "market_outlook": "Neutral",
            "key_factors": ["Fondprestation", "Ekonomiska indikatorer"],
            "source": "quantitative_analysis"
        }
    
    def _basic_rebalancing_advice(
        self,
        current: Dict[str, float],
        target: Dict[str, float]
    ) -> Dict[str, Any]:
        """Basic rebalancing advice fallback."""
        
        changes_needed = []
        for fund_id, target_weight in target.items():
            current_weight = current.get(fund_id, 0)
            if abs(target_weight - current_weight) > 0.05:  # 5% threshold
                direction = "öka" if target_weight > current_weight else "minska"
                changes_needed.append(f"{direction} {fund_id}")
        
        return {
            "actions_needed": changes_needed,
            "timing_advice": "Rebalansera vid större avvikelser",
            "source": "quantitative_analysis"
        }
    
    def _basic_housing_analysis(
        self,
        housing_data: Dict[str, Any],
        investment_returns: Dict[str, float],
        personal_situation: Dict[str, Any]
    ) -> str:
        """Basic housing analysis fallback."""
        
        age = personal_situation.get("age", 35)
        savings = personal_situation.get("savings", 500000)
        
        if age < 30 and savings < 1000000:
            return "Överväg att fortsätta hyra och investera för att bygga kapital."
        elif age > 35 and savings > 1500000:
            return "Ekonomiskt läge kan vara lämpligt för bostadsköp."
        else:
            return "Avväg bostadsköp mot investeringar baserat på personlig situation."
    
    def _basic_tax_advice(self, portfolio: Dict[str, float], realized_gains: float) -> str:
        """Basic tax advice fallback."""
        
        advice = "Överväg ISK-konto för förutsägbar beskattning. "
        
        if realized_gains > 0:
            advice += "Planera försäljningar för att optimera kapitalvinstskatt."
        
        return advice
    
    def _extract_housing_recommendation(self, response: str) -> str:
        """Extract housing recommendation from response."""
        
        response_lower = response.lower()
        
        if "köpa" in response_lower and "bostad" in response_lower:
            return "buy"
        elif "hyra" in response_lower:
            return "rent"
        else:
            return "evaluate"
    
    def _extract_tax_strategies(self, response: str) -> List[str]:
        """Extract tax strategies from response."""
        
        strategies = []
        
        if "isk" in response.lower():
            strategies.append("Använd ISK-konto")
        
        if "kapitalförlust" in response.lower():
            strategies.append("Realisera kapitalförluster")
        
        if "timing" in response.lower() or "tidpunkt" in response.lower():
            strategies.append("Planera försäljningstidpunkt")
        
        return strategies
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get AI engine system status."""
        
        try:
            gemma_available = await self.gemma.check_availability()
            
            return {
                "ai_engine_status": "operational",
                "gemma_available": gemma_available,
                "model_name": self.gemma.model_name,
                "cache_entries": len(self._analysis_cache),
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "ai_engine_status": "error",
                "error": str(e),
                "gemma_available": False,
                "last_check": datetime.now().isoformat()
            }