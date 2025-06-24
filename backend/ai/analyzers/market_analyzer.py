"""AI-powered market analysis with technical and fundamental insights."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..config import AIConfig
from ...analysis.patterns import TechnicalAnalyzer
from ...analysis.correlations import CorrelationAnalyzer
from ...analysis.risk_metrics import RiskAnalyzer

logger = logging.getLogger(__name__)


class MarketAnalyzer:
    """AI-powered market analysis combining technical analysis with AI insights."""
    
    def __init__(self, ai_config: AIConfig):
        """
        Initialize the market analyzer.
        
        Args:
            ai_config: AI configuration for provider access
        """
        self.ai_config = ai_config
        self.technical_analyzer = TechnicalAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def analyze_stock(
        self, 
        symbol: str, 
        market_data: List[Dict[str, Any]], 
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive AI-powered stock analysis.
        
        Args:
            symbol: Stock symbol to analyze
            market_data: Historical market data
            provider_name: AI provider to use
            
        Returns:
            Complete stock analysis with AI insights
        """
        try:
            self.logger.info(f"Starting comprehensive analysis for {symbol}")
            
            # Technical analysis
            technical_analysis = self.technical_analyzer.analyze_symbol(symbol, market_data)
            
            # Risk analysis
            volatility_metrics = self.risk_analyzer.calculate_volatility(market_data)
            drawdown_analysis = self.risk_analyzer.maximum_drawdown(market_data)
            var_analysis = self.risk_analyzer.value_at_risk(market_data)
            
            # Combine technical and risk data for AI analysis
            analysis_context = {
                "symbol": symbol,
                "technical_indicators": technical_analysis.get("moving_averages", {}),
                "trend_analysis": technical_analysis.get("trend_analysis", {}),
                "rsi": technical_analysis.get("rsi"),
                "support_resistance": technical_analysis.get("support_resistance", {}),
                "volatility": volatility_metrics,
                "risk_metrics": {
                    "max_drawdown": drawdown_analysis.get("maximum_drawdown"),
                    "var_5pct": var_analysis.get("historical_var"),
                    "current_drawdown": drawdown_analysis.get("current_drawdown")
                },
                "data_points": len(market_data)
            }
            
            # Get AI insights
            ai_insights = await self._get_ai_insights(symbol, analysis_context, provider_name)
            
            # Generate trading signals
            trading_signals = self._generate_trading_signals(technical_analysis, volatility_metrics)
            
            # Create comprehensive analysis result
            result = {
                "symbol": symbol,
                "analysis_timestamp": datetime.now().isoformat(),
                "technical_analysis": technical_analysis,
                "risk_analysis": {
                    "volatility": volatility_metrics,
                    "maximum_drawdown": drawdown_analysis,
                    "value_at_risk": var_analysis
                },
                "ai_insights": ai_insights,
                "trading_signals": trading_signals,
                "overall_assessment": self._create_overall_assessment(
                    technical_analysis, ai_insights, trading_signals
                )
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing stock {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat()
            }
    
    async def compare_stocks(
        self, 
        symbols_data: Dict[str, List[Dict[str, Any]]], 
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple stocks using AI analysis.
        
        Args:
            symbols_data: Dictionary of symbol -> market data
            provider_name: AI provider to use
            
        Returns:
            Comparative analysis results
        """
        try:
            self.logger.info(f"Comparing stocks: {list(symbols_data.keys())}")
            
            # Analyze each stock individually
            individual_analyses = {}
            for symbol, data in symbols_data.items():
                analysis = await self.analyze_stock(symbol, data, provider_name)
                individual_analyses[symbol] = analysis
            
            # Calculate correlations between stocks
            correlation_analysis = self.correlation_analyzer.calculate_portfolio_correlations(symbols_data)
            
            # Extract key metrics for comparison
            comparison_metrics = {}
            for symbol, analysis in individual_analyses.items():
                tech_score = analysis.get("technical_analysis", {}).get("technical_score", {})
                risk_metrics = analysis.get("risk_analysis", {})
                
                comparison_metrics[symbol] = {
                    "technical_score": tech_score.get("score", 0),
                    "signal": tech_score.get("signal", "neutral"),
                    "volatility": risk_metrics.get("volatility", {}).get("historical_volatility", 0),
                    "max_drawdown": risk_metrics.get("maximum_drawdown", {}).get("maximum_drawdown", 0),
                    "rsi": analysis.get("technical_analysis", {}).get("rsi"),
                    "trend": analysis.get("technical_analysis", {}).get("trend_analysis", {}).get("trend", "neutral")
                }
            
            # Get AI comparison insights
            ai_comparison = await self._get_comparative_ai_insights(
                comparison_metrics, correlation_analysis, provider_name
            )
            
            # Generate ranking
            ranking = self._rank_stocks(comparison_metrics)
            
            return {
                "comparison_timestamp": datetime.now().isoformat(),
                "symbols_analyzed": list(symbols_data.keys()),
                "individual_analyses": individual_analyses,
                "correlation_analysis": correlation_analysis,
                "comparison_metrics": comparison_metrics,
                "ai_comparison": ai_comparison,
                "ranking": ranking,
                "recommendation": self._generate_portfolio_recommendation(ranking, correlation_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing stocks: {e}")
            return {"error": str(e)}
    
    async def market_sentiment_analysis(
        self, 
        market_data: Dict[str, List[Dict[str, Any]]], 
        economic_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze overall market sentiment using AI.
        
        Args:
            market_data: Market data for multiple symbols
            economic_data: Optional economic indicators
            provider_name: AI provider to use
            
        Returns:
            Market sentiment analysis
        """
        try:
            self.logger.info("Analyzing market sentiment")
            
            # Calculate market-wide metrics
            market_metrics = {}
            for symbol, data in market_data.items():
                if data:
                    tech_analysis = self.technical_analyzer.analyze_symbol(symbol, data)
                    vol_metrics = self.risk_analyzer.calculate_volatility(data)
                    
                    market_metrics[symbol] = {
                        "trend": tech_analysis.get("trend_analysis", {}).get("trend", "neutral"),
                        "trend_strength": tech_analysis.get("trend_analysis", {}).get("strength", 0),
                        "rsi": tech_analysis.get("rsi"),
                        "volatility": vol_metrics.get("historical_volatility", 0),
                        "technical_score": tech_analysis.get("technical_score", {}).get("score", 0)
                    }
            
            # Calculate aggregate sentiment metrics
            sentiment_metrics = self._calculate_sentiment_metrics(market_metrics)
            
            # Include economic correlations if available
            economic_correlations = {}
            if economic_data:
                for symbol, mkt_data in market_data.items():
                    for econ_type, econ_data in economic_data.items():
                        if mkt_data and econ_data:
                            corr_key = f"{symbol}_vs_{econ_type}"
                            if econ_type == "interest_rates":
                                corr_analysis = self.correlation_analyzer.analyze_rates_vs_stocks(
                                    mkt_data, econ_data, symbol
                                )
                            else:
                                corr_analysis = self.correlation_analyzer.analyze_currency_impact(
                                    mkt_data, econ_data
                                )
                            economic_correlations[corr_key] = corr_analysis
            
            # Get AI sentiment insights
            ai_sentiment = await self._get_sentiment_ai_insights(
                sentiment_metrics, economic_correlations, provider_name
            )
            
            return {
                "analysis_timestamp": datetime.now().isoformat(),
                "market_metrics": market_metrics,
                "sentiment_metrics": sentiment_metrics,
                "economic_correlations": economic_correlations,
                "ai_sentiment": ai_sentiment,
                "market_outlook": self._generate_market_outlook(sentiment_metrics, ai_sentiment)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market sentiment: {e}")
            return {"error": str(e)}
    
    async def _get_ai_insights(
        self, 
        symbol: str, 
        context: Dict[str, Any], 
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get AI insights for individual stock analysis."""
        try:
            provider = self.ai_config.get_provider(provider_name)
            
            # Prepare market data for AI analysis
            market_data_summary = {
                "current_price": context.get("support_resistance", {}).get("current_price"),
                "volume": "N/A",  # Would need to be extracted from raw data
                "close_price": context.get("support_resistance", {}).get("current_price")
            }
            
            if hasattr(provider, '__aenter__'):
                async with provider:
                    insights = await provider.analyze_market_data(
                        symbol, market_data_summary, context.get("technical_indicators", {})
                    )
            else:
                insights = await provider.analyze_market_data(
                    symbol, market_data_summary, context.get("technical_indicators", {})
                )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting AI insights for {symbol}: {e}")
            return {"error": str(e), "insights": "AI analysis unavailable"}
    
    async def _get_comparative_ai_insights(
        self, 
        metrics: Dict[str, Dict[str, Any]], 
        correlation_data: Dict[str, Any],
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get AI insights for stock comparison."""
        try:
            provider = self.ai_config.get_provider(provider_name)
            
            context = {
                "comparison_metrics": metrics,
                "correlation_analysis": correlation_data.get("diversification_metrics", {}),
                "symbols": list(metrics.keys())
            }
            
            if hasattr(provider, '__aenter__'):
                async with provider:
                    insights = await provider.generate_insights(context, "comparative_stock")
            else:
                insights = await provider.generate_insights(context, "comparative_stock")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting comparative AI insights: {e}")
            return {"error": str(e)}
    
    async def _get_sentiment_ai_insights(
        self, 
        sentiment_metrics: Dict[str, Any],
        economic_correlations: Dict[str, Any],
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get AI insights for market sentiment."""
        try:
            provider = self.ai_config.get_provider(provider_name)
            
            context = {
                "market_sentiment": sentiment_metrics,
                "economic_correlations": economic_correlations,
                "analysis_type": "market_sentiment"
            }
            
            if hasattr(provider, '__aenter__'):
                async with provider:
                    insights = await provider.generate_insights(context, "market_sentiment")
            else:
                insights = await provider.generate_insights(context, "market_sentiment")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment AI insights: {e}")
            return {"error": str(e)}
    
    def _generate_trading_signals(
        self, 
        technical_analysis: Dict[str, Any], 
        volatility_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate trading signals based on analysis."""
        signals = {
            "entry_signals": [],
            "exit_signals": [],
            "risk_warnings": [],
            "overall_signal": "neutral"
        }
        
        try:
            # Technical signals
            tech_score = technical_analysis.get("technical_score", {})
            if tech_score.get("signal") == "bullish" and tech_score.get("score", 0) > 60:
                signals["entry_signals"].append("Strong bullish technical indicators")
                signals["overall_signal"] = "buy"
            elif tech_score.get("signal") == "bearish" and tech_score.get("score", 0) < -60:
                signals["exit_signals"].append("Strong bearish technical indicators")
                signals["overall_signal"] = "sell"
            
            # RSI signals
            rsi = technical_analysis.get("rsi")
            if rsi:
                if rsi < 30:
                    signals["entry_signals"].append(f"RSI oversold at {rsi:.1f}")
                elif rsi > 70:
                    signals["exit_signals"].append(f"RSI overbought at {rsi:.1f}")
            
            # Volatility warnings
            vol_class = volatility_metrics.get("volatility_class")
            if vol_class in ["high", "very high"]:
                signals["risk_warnings"].append(f"High volatility detected ({vol_class})")
            
            # Support/resistance signals
            sr = technical_analysis.get("support_resistance", {})
            if sr.get("near_support"):
                signals["entry_signals"].append("Price near support level")
            elif sr.get("near_resistance"):
                signals["exit_signals"].append("Price near resistance level")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {e}")
            return signals
    
    def _create_overall_assessment(
        self, 
        technical_analysis: Dict[str, Any], 
        ai_insights: Dict[str, Any], 
        trading_signals: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create overall investment assessment."""
        try:
            # Combine technical score and AI assessment
            tech_score = technical_analysis.get("technical_score", {}).get("score", 0)
            ai_assessment = ai_insights.get("assessment", "neutral")
            
            # Map AI assessment to numeric score
            ai_score_map = {"bullish": 70, "bearish": -70, "neutral": 0}
            ai_score = ai_score_map.get(ai_assessment, 0)
            
            # Weighted average (60% technical, 40% AI)
            combined_score = (tech_score * 0.6) + (ai_score * 0.4)
            
            # Determine recommendation
            if combined_score > 50:
                recommendation = "buy"
                confidence = "high" if combined_score > 70 else "medium"
            elif combined_score < -50:
                recommendation = "sell"
                confidence = "high" if combined_score < -70 else "medium"
            else:
                recommendation = "hold"
                confidence = "medium"
            
            # Risk level assessment
            risk_factors = len(trading_signals.get("risk_warnings", []))
            if risk_factors > 2:
                risk_level = "high"
            elif risk_factors > 0:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return {
                "recommendation": recommendation,
                "confidence": confidence,
                "combined_score": round(combined_score, 1),
                "technical_score": tech_score,
                "ai_score": ai_score,
                "risk_level": risk_level,
                "key_factors": {
                    "entry_signals": trading_signals.get("entry_signals", []),
                    "exit_signals": trading_signals.get("exit_signals", []),
                    "risk_warnings": trading_signals.get("risk_warnings", [])
                },
                "ai_reasoning": ai_insights.get("reasoning", "No AI reasoning available")
            }
            
        except Exception as e:
            self.logger.error(f"Error creating overall assessment: {e}")
            return {"error": str(e)}
    
    def _calculate_sentiment_metrics(self, market_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate market sentiment metrics."""
        try:
            if not market_metrics:
                return {}
            
            # Count trends
            bullish_count = sum(1 for metrics in market_metrics.values() 
                              if metrics.get("trend") == "bullish")
            bearish_count = sum(1 for metrics in market_metrics.values() 
                               if metrics.get("trend") == "bearish")
            neutral_count = len(market_metrics) - bullish_count - bearish_count
            
            # Average technical scores
            tech_scores = [metrics.get("technical_score", 0) for metrics in market_metrics.values()]
            avg_tech_score = sum(tech_scores) / len(tech_scores) if tech_scores else 0
            
            # Average volatility
            volatilities = [metrics.get("volatility", 0) for metrics in market_metrics.values()]
            avg_volatility = sum(volatilities) / len(volatilities) if volatilities else 0
            
            # Overall sentiment
            if avg_tech_score > 30:
                overall_sentiment = "bullish"
            elif avg_tech_score < -30:
                overall_sentiment = "bearish"
            else:
                overall_sentiment = "neutral"
            
            return {
                "overall_sentiment": overall_sentiment,
                "average_technical_score": round(avg_tech_score, 1),
                "average_volatility": round(avg_volatility, 3),
                "trend_distribution": {
                    "bullish": bullish_count,
                    "bearish": bearish_count,
                    "neutral": neutral_count
                },
                "market_stress": "high" if avg_volatility > 0.3 else "medium" if avg_volatility > 0.2 else "low"
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating sentiment metrics: {e}")
            return {}
    
    def _rank_stocks(self, metrics: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank stocks based on multiple criteria."""
        try:
            ranked_stocks = []
            
            for symbol, data in metrics.items():
                score = data.get("technical_score", 0)
                volatility = data.get("volatility", 0)
                max_dd = abs(data.get("max_drawdown", 0))
                
                # Risk-adjusted score (penalize high volatility and drawdown)
                risk_penalty = (volatility * 100) + (max_dd * 50)
                adjusted_score = score - risk_penalty
                
                ranked_stocks.append({
                    "symbol": symbol,
                    "score": score,
                    "adjusted_score": round(adjusted_score, 1),
                    "signal": data.get("signal", "neutral"),
                    "volatility": volatility,
                    "max_drawdown": max_dd,
                    "trend": data.get("trend", "neutral")
                })
            
            # Sort by adjusted score (highest first)
            ranked_stocks.sort(key=lambda x: x["adjusted_score"], reverse=True)
            
            return ranked_stocks
            
        except Exception as e:
            self.logger.error(f"Error ranking stocks: {e}")
            return []
    
    def _generate_portfolio_recommendation(
        self, 
        ranking: List[Dict[str, Any]], 
        correlation_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate portfolio construction recommendation."""
        try:
            if not ranking:
                return {"error": "No ranking data available"}
            
            # Get diversification metrics
            diversification = correlation_analysis.get("diversification_metrics", {})
            div_level = diversification.get("diversification_level", "unknown")
            
            # Top picks (positive adjusted scores)
            top_picks = [stock for stock in ranking if stock["adjusted_score"] > 0][:3]
            avoid_picks = [stock for stock in ranking if stock["adjusted_score"] < -30]
            
            # Portfolio allocation suggestion
            if len(top_picks) > 1 and div_level in ["good", "excellent"]:
                allocation_strategy = "balanced_diversified"
                recommendation = "Build diversified portfolio with top-ranked stocks"
            elif len(top_picks) == 1:
                allocation_strategy = "concentrated"
                recommendation = "Focus on single top performer with caution"
            else:
                allocation_strategy = "defensive"
                recommendation = "Consider defensive positioning or cash"
            
            return {
                "allocation_strategy": allocation_strategy,
                "recommendation": recommendation,
                "top_picks": top_picks,
                "avoid_picks": avoid_picks,
                "diversification_level": div_level,
                "risk_assessment": "high" if len(avoid_picks) > len(top_picks) else "moderate"
            }
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio recommendation: {e}")
            return {"error": str(e)}
    
    def _generate_market_outlook(
        self, 
        sentiment_metrics: Dict[str, Any], 
        ai_sentiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate overall market outlook."""
        try:
            overall_sentiment = sentiment_metrics.get("overall_sentiment", "neutral")
            avg_score = sentiment_metrics.get("average_technical_score", 0)
            market_stress = sentiment_metrics.get("market_stress", "medium")
            
            # Generate outlook based on metrics
            if overall_sentiment == "bullish" and avg_score > 40:
                outlook = "positive"
                timeframe = "short_to_medium_term"
            elif overall_sentiment == "bearish" and avg_score < -40:
                outlook = "negative" 
                timeframe = "short_to_medium_term"
            else:
                outlook = "mixed"
                timeframe = "uncertain"
            
            # Risk considerations
            risks = []
            if market_stress == "high":
                risks.append("High market volatility")
            if avg_score < -20:
                risks.append("Weak technical momentum")
            
            opportunities = []
            if overall_sentiment == "bullish":
                opportunities.append("Positive technical momentum")
            if market_stress == "low":
                opportunities.append("Low volatility environment")
            
            return {
                "outlook": outlook,
                "timeframe": timeframe,
                "confidence": ai_sentiment.get("confidence", 0.5),
                "key_risks": risks,
                "key_opportunities": opportunities,
                "recommended_strategy": self._get_strategy_recommendation(outlook, market_stress)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating market outlook: {e}")
            return {"error": str(e)}
    
    def _get_strategy_recommendation(self, outlook: str, market_stress: str) -> str:
        """Get investment strategy recommendation."""
        if outlook == "positive" and market_stress == "low":
            return "Growth-oriented with moderate risk"
        elif outlook == "positive" and market_stress == "high":
            return "Selective growth with risk management"
        elif outlook == "negative":
            return "Defensive positioning with cash allocation"
        else:
            return "Balanced approach with diversification"