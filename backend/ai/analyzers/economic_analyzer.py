"""AI-powered economic analysis with policy and macro insights."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from ..config import AIConfig
from ...analysis.correlations import CorrelationAnalyzer

logger = logging.getLogger(__name__)


class EconomicAnalyzer:
    """AI-powered economic and policy analysis."""
    
    def __init__(self, ai_config: AIConfig):
        """
        Initialize the economic analyzer.
        
        Args:
            ai_config: AI configuration for provider access
        """
        self.ai_config = ai_config
        self.correlation_analyzer = CorrelationAnalyzer()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def analyze_monetary_policy(
        self, 
        interest_rate_data: List[Dict[str, Any]], 
        market_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze monetary policy impact using AI.
        
        Args:
            interest_rate_data: Central bank interest rate data
            market_data: Optional market data for correlation analysis
            provider_name: AI provider to use
            
        Returns:
            Monetary policy analysis
        """
        try:
            self.logger.info("Analyzing monetary policy impact")
            
            # Analyze interest rate trends
            rate_analysis = self._analyze_rate_trends(interest_rate_data)
            
            # Calculate market correlations if market data available
            market_correlations = {}
            if market_data:
                for symbol, data in market_data.items():
                    correlation = self.correlation_analyzer.analyze_rates_vs_stocks(
                        data, interest_rate_data, symbol
                    )
                    market_correlations[symbol] = correlation
            
            # Get AI insights on monetary policy
            ai_insights = await self._get_monetary_policy_insights(
                rate_analysis, market_correlations, provider_name
            )
            
            # Generate policy implications
            implications = self._generate_policy_implications(rate_analysis, ai_insights)
            
            return {
                "analysis_timestamp": datetime.now().isoformat(),
                "rate_analysis": rate_analysis,
                "market_correlations": market_correlations,
                "ai_insights": ai_insights,
                "policy_implications": implications,
                "investment_recommendations": self._generate_rate_investment_recommendations(
                    rate_analysis, market_correlations
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing monetary policy: {e}")
            return {"error": str(e)}
    
    async def analyze_economic_indicators(
        self, 
        housing_data: Optional[List[Dict[str, Any]]] = None,
        employment_data: Optional[List[Dict[str, Any]]] = None,
        inflation_data: Optional[List[Dict[str, Any]]] = None,
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze key economic indicators using AI.
        
        Args:
            housing_data: Housing price index data
            employment_data: Employment statistics
            inflation_data: Inflation rate data
            provider_name: AI provider to use
            
        Returns:
            Economic indicators analysis
        """
        try:
            self.logger.info("Analyzing economic indicators")
            
            # Analyze each indicator
            indicator_analysis = {}
            
            if housing_data:
                indicator_analysis["housing"] = self._analyze_housing_trends(housing_data)
            
            if employment_data:
                indicator_analysis["employment"] = self._analyze_employment_trends(employment_data)
            
            if inflation_data:
                indicator_analysis["inflation"] = self._analyze_inflation_trends(inflation_data)
            
            # Calculate cross-correlations between indicators
            cross_correlations = self._calculate_indicator_correlations(
                housing_data, employment_data, inflation_data
            )
            
            # Get AI insights on economic health
            ai_insights = await self._get_economic_health_insights(
                indicator_analysis, cross_correlations, provider_name
            )
            
            # Generate economic outlook
            outlook = self._generate_economic_outlook(indicator_analysis, ai_insights)
            
            return {
                "analysis_timestamp": datetime.now().isoformat(),
                "indicator_analysis": indicator_analysis,
                "cross_correlations": cross_correlations,
                "ai_insights": ai_insights,
                "economic_outlook": outlook,
                "policy_predictions": self._predict_policy_responses(indicator_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing economic indicators: {e}")
            return {"error": str(e)}
    
    async def analyze_currency_impact(
        self, 
        currency_data: List[Dict[str, Any]], 
        market_data: Dict[str, List[Dict[str, Any]]],
        currency_pair: str = "SEK/USD",
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze currency impact on markets using AI.
        
        Args:
            currency_data: Currency exchange rate data
            market_data: Market data for correlation analysis
            currency_pair: Currency pair description
            provider_name: AI provider to use
            
        Returns:
            Currency impact analysis
        """
        try:
            self.logger.info(f"Analyzing currency impact for {currency_pair}")
            
            # Analyze currency trends
            currency_analysis = self._analyze_currency_trends(currency_data, currency_pair)
            
            # Calculate market correlations
            market_correlations = {}
            for symbol, data in market_data.items():
                correlation = self.correlation_analyzer.analyze_currency_impact(
                    data, currency_data, currency_pair
                )
                market_correlations[symbol] = correlation
            
            # Get AI insights on currency impact
            ai_insights = await self._get_currency_impact_insights(
                currency_analysis, market_correlations, currency_pair, provider_name
            )
            
            # Generate trading implications
            trading_implications = self._generate_currency_trading_implications(
                currency_analysis, market_correlations, ai_insights
            )
            
            return {
                "analysis_timestamp": datetime.now().isoformat(),
                "currency_pair": currency_pair,
                "currency_analysis": currency_analysis,
                "market_correlations": market_correlations,
                "ai_insights": ai_insights,
                "trading_implications": trading_implications,
                "hedging_recommendations": self._generate_hedging_recommendations(
                    currency_analysis, market_correlations
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing currency impact: {e}")
            return {"error": str(e)}
    
    async def comprehensive_macro_analysis(
        self, 
        economic_data: Dict[str, List[Dict[str, Any]]],
        market_data: Dict[str, List[Dict[str, Any]]],
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive macroeconomic analysis.
        
        Args:
            economic_data: Dictionary of economic indicator data
            market_data: Market data for various symbols
            provider_name: AI provider to use
            
        Returns:
            Comprehensive macro analysis
        """
        try:
            self.logger.info("Performing comprehensive macro analysis")
            
            # Analyze individual economic components
            components = {}
            
            if "interest_rates" in economic_data:
                components["monetary_policy"] = await self.analyze_monetary_policy(
                    economic_data["interest_rates"], market_data, provider_name
                )
            
            if any(key in economic_data for key in ["housing", "employment", "inflation"]):
                components["economic_indicators"] = await self.analyze_economic_indicators(
                    economic_data.get("housing"),
                    economic_data.get("employment"), 
                    economic_data.get("inflation"),
                    provider_name
                )
            
            if "currency" in economic_data:
                components["currency_impact"] = await self.analyze_currency_impact(
                    economic_data["currency"], market_data, "SEK/USD", provider_name
                )
            
            # Calculate comprehensive correlations
            comprehensive_correlations = self._calculate_comprehensive_correlations(
                economic_data, market_data
            )
            
            # Get AI macro insights
            ai_macro_insights = await self._get_macro_insights(
                components, comprehensive_correlations, provider_name
            )
            
            # Generate macro outlook and recommendations
            macro_outlook = self._generate_macro_outlook(components, ai_macro_insights)
            
            return {
                "analysis_timestamp": datetime.now().isoformat(),
                "components": components,
                "comprehensive_correlations": comprehensive_correlations,
                "ai_macro_insights": ai_macro_insights,
                "macro_outlook": macro_outlook,
                "strategic_recommendations": self._generate_strategic_recommendations(
                    macro_outlook, components
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive macro analysis: {e}")
            return {"error": str(e)}
    
    def _analyze_rate_trends(self, rate_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze interest rate trends and patterns."""
        try:
            if not rate_data:
                return {"error": "No rate data available"}
            
            # Sort by date
            sorted_data = sorted(rate_data, key=lambda x: x['date'])
            rates = [float(item['value']) for item in sorted_data]
            
            if len(rates) < 2:
                return {"error": "Insufficient data for trend analysis"}
            
            # Calculate trend
            current_rate = rates[-1]
            previous_rate = rates[-2] if len(rates) > 1 else current_rate
            rate_change = current_rate - previous_rate
            
            # Calculate longer-term trend
            if len(rates) >= 12:
                year_ago_rate = rates[-12]
                annual_change = current_rate - year_ago_rate
            else:
                annual_change = rate_change
            
            # Determine trend direction
            if rate_change > 0.1:
                short_term_trend = "tightening"
            elif rate_change < -0.1:
                short_term_trend = "easing"
            else:
                short_term_trend = "stable"
            
            # Rate level assessment
            if current_rate < 1.0:
                rate_level = "accommodative"
            elif current_rate < 3.0:
                rate_level = "neutral"
            else:
                rate_level = "restrictive"
            
            return {
                "current_rate": current_rate,
                "rate_change": rate_change,
                "annual_change": annual_change,
                "short_term_trend": short_term_trend,
                "rate_level": rate_level,
                "data_points": len(rates),
                "volatility": self._calculate_rate_volatility(rates)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing rate trends: {e}")
            return {"error": str(e)}
    
    def _analyze_housing_trends(self, housing_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze housing market trends."""
        try:
            if not housing_data:
                return {"error": "No housing data available"}
            
            sorted_data = sorted(housing_data, key=lambda x: x['date'])
            values = [float(item['value']) for item in sorted_data]
            
            if len(values) < 2:
                return {"error": "Insufficient housing data"}
            
            # Calculate growth rates
            current_value = values[-1]
            if len(values) >= 12:
                year_ago_value = values[-12]
                annual_growth = ((current_value - year_ago_value) / year_ago_value) * 100
            else:
                annual_growth = 0
            
            # Quarter-over-quarter growth
            if len(values) >= 3:
                quarter_ago_value = values[-3]
                quarterly_growth = ((current_value - quarter_ago_value) / quarter_ago_value) * 100
            else:
                quarterly_growth = 0
            
            # Trend assessment
            if annual_growth > 5:
                trend = "strong_growth"
            elif annual_growth > 0:
                trend = "moderate_growth"
            elif annual_growth > -5:
                trend = "declining"
            else:
                trend = "sharp_decline"
            
            return {
                "current_index": current_value,
                "annual_growth": annual_growth,
                "quarterly_growth": quarterly_growth,
                "trend": trend,
                "data_points": len(values)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing housing trends: {e}")
            return {"error": str(e)}
    
    def _analyze_employment_trends(self, employment_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze employment trends."""
        try:
            if not employment_data:
                return {"error": "No employment data available"}
            
            sorted_data = sorted(employment_data, key=lambda x: x['date'])
            values = [float(item['value']) for item in sorted_data]
            
            current_rate = values[-1]
            
            # Determine employment health
            if current_rate < 4:
                employment_health = "very_strong"
            elif current_rate < 6:
                employment_health = "strong"
            elif current_rate < 8:
                employment_health = "moderate"
            else:
                employment_health = "weak"
            
            # Calculate trend
            if len(values) >= 3:
                recent_trend = values[-1] - values[-3]
                if recent_trend < -0.5:
                    trend = "improving"
                elif recent_trend > 0.5:
                    trend = "deteriorating"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"
            
            return {
                "current_rate": current_rate,
                "employment_health": employment_health,
                "trend": trend,
                "data_points": len(values)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing employment trends: {e}")
            return {"error": str(e)}
    
    def _analyze_inflation_trends(self, inflation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze inflation trends."""
        try:
            if not inflation_data:
                return {"error": "No inflation data available"}
            
            sorted_data = sorted(inflation_data, key=lambda x: x['date'])
            values = [float(item['value']) for item in sorted_data]
            
            current_rate = values[-1]
            
            # Inflation assessment
            if current_rate < 1:
                inflation_level = "deflationary"
            elif current_rate < 2:
                inflation_level = "low"
            elif current_rate < 4:
                inflation_level = "moderate"
            else:
                inflation_level = "high"
            
            # Calculate trend
            if len(values) >= 6:
                six_month_avg = sum(values[-6:]) / 6
                if current_rate > six_month_avg * 1.1:
                    trend = "accelerating"
                elif current_rate < six_month_avg * 0.9:
                    trend = "decelerating"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"
            
            return {
                "current_rate": current_rate,
                "inflation_level": inflation_level,
                "trend": trend,
                "data_points": len(values)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing inflation trends: {e}")
            return {"error": str(e)}
    
    def _analyze_currency_trends(self, currency_data: List[Dict[str, Any]], pair: str) -> Dict[str, Any]:
        """Analyze currency exchange rate trends."""
        try:
            if not currency_data:
                return {"error": "No currency data available"}
            
            sorted_data = sorted(currency_data, key=lambda x: x['date'])
            rates = [float(item['value']) for item in sorted_data]
            
            current_rate = rates[-1]
            
            # Calculate changes
            if len(rates) >= 30:
                month_ago_rate = rates[-30]
                monthly_change = ((current_rate - month_ago_rate) / month_ago_rate) * 100
            else:
                monthly_change = 0
            
            # Volatility
            if len(rates) >= 20:
                recent_rates = rates[-20:]
                volatility = (max(recent_rates) - min(recent_rates)) / min(recent_rates) * 100
            else:
                volatility = 0
            
            # Trend assessment
            if monthly_change > 2:
                trend = "strengthening" if "SEK" in pair else "weakening"
            elif monthly_change < -2:
                trend = "weakening" if "SEK" in pair else "strengthening"
            else:
                trend = "stable"
            
            return {
                "current_rate": current_rate,
                "monthly_change": monthly_change,
                "trend": trend,
                "volatility": volatility,
                "data_points": len(rates)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing currency trends: {e}")
            return {"error": str(e)}
    
    async def _get_monetary_policy_insights(
        self, 
        rate_analysis: Dict[str, Any], 
        market_correlations: Dict[str, Any],
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get AI insights on monetary policy."""
        try:
            provider = self.ai_config.get_provider(provider_name)
            
            context = {
                "interest_rate_analysis": rate_analysis,
                "market_correlations": market_correlations,
                "analysis_type": "monetary_policy"
            }
            
            if hasattr(provider, '__aenter__'):
                async with provider:
                    insights = await provider.generate_insights(context, "monetary_policy")
            else:
                insights = await provider.generate_insights(context, "monetary_policy")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting monetary policy insights: {e}")
            return {"error": str(e)}
    
    async def _get_economic_health_insights(
        self, 
        indicator_analysis: Dict[str, Any], 
        cross_correlations: Dict[str, Any],
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get AI insights on economic health."""
        try:
            provider = self.ai_config.get_provider(provider_name)
            
            context = {
                "economic_indicators": indicator_analysis,
                "cross_correlations": cross_correlations,
                "analysis_type": "economic_health"
            }
            
            if hasattr(provider, '__aenter__'):
                async with provider:
                    insights = await provider.generate_insights(context, "economic_health")
            else:
                insights = await provider.generate_insights(context, "economic_health")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting economic health insights: {e}")
            return {"error": str(e)}
    
    async def _get_currency_impact_insights(
        self, 
        currency_analysis: Dict[str, Any], 
        market_correlations: Dict[str, Any], 
        currency_pair: str,
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get AI insights on currency impact."""
        try:
            provider = self.ai_config.get_provider(provider_name)
            
            context = {
                "currency_analysis": currency_analysis,
                "market_correlations": market_correlations,
                "currency_pair": currency_pair,
                "analysis_type": "currency_impact"
            }
            
            if hasattr(provider, '__aenter__'):
                async with provider:
                    insights = await provider.generate_insights(context, "currency_impact")
            else:
                insights = await provider.generate_insights(context, "currency_impact")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting currency impact insights: {e}")
            return {"error": str(e)}
    
    async def _get_macro_insights(
        self, 
        components: Dict[str, Any], 
        correlations: Dict[str, Any],
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive macro insights."""
        try:
            provider = self.ai_config.get_provider(provider_name)
            
            context = {
                "macro_components": components,
                "comprehensive_correlations": correlations,
                "analysis_type": "comprehensive_macro"
            }
            
            if hasattr(provider, '__aenter__'):
                async with provider:
                    insights = await provider.generate_insights(context, "macro_analysis")
            else:
                insights = await provider.generate_insights(context, "macro_analysis")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting macro insights: {e}")
            return {"error": str(e)}
    
    def _calculate_rate_volatility(self, rates: List[float]) -> float:
        """Calculate interest rate volatility."""
        if len(rates) < 2:
            return 0
        
        # Calculate standard deviation of rate changes
        changes = [rates[i] - rates[i-1] for i in range(1, len(rates))]
        mean_change = sum(changes) / len(changes)
        variance = sum((x - mean_change) ** 2 for x in changes) / len(changes)
        return variance ** 0.5
    
    def _calculate_indicator_correlations(
        self, 
        housing_data: Optional[List[Dict[str, Any]]], 
        employment_data: Optional[List[Dict[str, Any]]], 
        inflation_data: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Calculate correlations between economic indicators."""
        # This would need more sophisticated correlation analysis
        # For now, return placeholder
        return {
            "housing_employment": 0.0,
            "housing_inflation": 0.0,
            "employment_inflation": 0.0,
            "note": "Cross-correlations require more sophisticated analysis"
        }
    
    def _calculate_comprehensive_correlations(
        self, 
        economic_data: Dict[str, List[Dict[str, Any]]], 
        market_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Calculate comprehensive correlations across all data."""
        try:
            correlations = {}
            
            # Use existing correlation analyzer for market-economic correlations
            for econ_type, econ_data in economic_data.items():
                for market_symbol, mkt_data in market_data.items():
                    if econ_data and mkt_data:
                        key = f"{market_symbol}_vs_{econ_type}"
                        if econ_type == "interest_rates":
                            corr = self.correlation_analyzer.analyze_rates_vs_stocks(
                                mkt_data, econ_data, market_symbol
                            )
                        else:
                            corr = self.correlation_analyzer.analyze_currency_impact(
                                mkt_data, econ_data
                            )
                        correlations[key] = corr
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error calculating comprehensive correlations: {e}")
            return {}
    
    def _generate_policy_implications(
        self, 
        rate_analysis: Dict[str, Any], 
        ai_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate policy implications from rate analysis."""
        try:
            current_rate = rate_analysis.get("current_rate", 0)
            trend = rate_analysis.get("short_term_trend", "stable")
            level = rate_analysis.get("rate_level", "neutral")
            
            implications = []
            
            if trend == "tightening":
                implications.append("Continued monetary tightening expected")
                implications.append("Potential pressure on credit-sensitive sectors")
            elif trend == "easing":
                implications.append("Accommodative monetary policy supportive for growth")
                implications.append("Lower borrowing costs may stimulate investment")
            
            if level == "restrictive":
                implications.append("High rates may slow economic growth")
            elif level == "accommodative":
                implications.append("Low rates support economic expansion")
            
            return {
                "key_implications": implications,
                "policy_stance": level,
                "next_likely_move": self._predict_next_rate_move(rate_analysis),
                "market_impact": "negative" if trend == "tightening" else "positive"
            }
            
        except Exception as e:
            self.logger.error(f"Error generating policy implications: {e}")
            return {"error": str(e)}
    
    def _predict_next_rate_move(self, rate_analysis: Dict[str, Any]) -> str:
        """Predict next likely central bank move."""
        trend = rate_analysis.get("short_term_trend", "stable")
        level = rate_analysis.get("rate_level", "neutral")
        
        if trend == "tightening" and level != "restrictive":
            return "likely_hike"
        elif trend == "easing" and level != "accommodative":
            return "likely_cut"
        else:
            return "likely_hold"
    
    def _generate_rate_investment_recommendations(
        self, 
        rate_analysis: Dict[str, Any], 
        market_correlations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate investment recommendations based on rate environment."""
        try:
            trend = rate_analysis.get("short_term_trend", "stable")
            level = rate_analysis.get("rate_level", "neutral")
            
            recommendations = []
            sectors_favored = []
            sectors_avoid = []
            
            if trend == "tightening" or level == "restrictive":
                recommendations.append("Consider defensive positioning")
                sectors_favored.extend(["financials", "utilities"])
                sectors_avoid.extend(["real_estate", "growth_stocks"])
            elif trend == "easing" or level == "accommodative":
                recommendations.append("Growth stocks may benefit")
                sectors_favored.extend(["technology", "real_estate"])
                sectors_avoid.extend(["cash", "short_term_bonds"])
            
            return {
                "investment_recommendations": recommendations,
                "sectors_favored": sectors_favored,
                "sectors_to_avoid": sectors_avoid,
                "duration_strategy": "short" if trend == "tightening" else "long",
                "risk_assessment": "elevated" if trend == "tightening" else "moderate"
            }
            
        except Exception as e:
            self.logger.error(f"Error generating rate investment recommendations: {e}")
            return {"error": str(e)}
    
    def _generate_economic_outlook(
        self, 
        indicator_analysis: Dict[str, Any], 
        ai_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate economic outlook from indicators."""
        try:
            # Assess overall economic health
            health_scores = []
            
            housing = indicator_analysis.get("housing", {})
            if housing and "trend" in housing:
                if housing["trend"] in ["strong_growth", "moderate_growth"]:
                    health_scores.append(1)
                elif housing["trend"] == "declining":
                    health_scores.append(-1)
                else:
                    health_scores.append(-2)
            
            employment = indicator_analysis.get("employment", {})
            if employment and "employment_health" in employment:
                health_map = {"very_strong": 2, "strong": 1, "moderate": 0, "weak": -2}
                health_scores.append(health_map.get(employment["employment_health"], 0))
            
            # Overall assessment
            avg_score = sum(health_scores) / len(health_scores) if health_scores else 0
            
            if avg_score > 0.5:
                outlook = "positive"
            elif avg_score < -0.5:
                outlook = "negative"
            else:
                outlook = "mixed"
            
            return {
                "outlook": outlook,
                "confidence": abs(avg_score),
                "key_strengths": self._identify_economic_strengths(indicator_analysis),
                "key_weaknesses": self._identify_economic_weaknesses(indicator_analysis),
                "timeline": "6_to_12_months"
            }
            
        except Exception as e:
            self.logger.error(f"Error generating economic outlook: {e}")
            return {"error": str(e)}
    
    def _identify_economic_strengths(self, indicator_analysis: Dict[str, Any]) -> List[str]:
        """Identify economic strengths from indicators."""
        strengths = []
        
        housing = indicator_analysis.get("housing", {})
        if housing.get("trend") in ["strong_growth", "moderate_growth"]:
            strengths.append("Healthy housing market growth")
        
        employment = indicator_analysis.get("employment", {})
        if employment.get("employment_health") in ["very_strong", "strong"]:
            strengths.append("Strong employment conditions")
        
        return strengths
    
    def _identify_economic_weaknesses(self, indicator_analysis: Dict[str, Any]) -> List[str]:
        """Identify economic weaknesses from indicators."""
        weaknesses = []
        
        housing = indicator_analysis.get("housing", {})
        if housing.get("trend") in ["declining", "sharp_decline"]:
            weaknesses.append("Declining housing market")
        
        employment = indicator_analysis.get("employment", {})
        if employment.get("employment_health") == "weak":
            weaknesses.append("Weak employment conditions")
        
        inflation = indicator_analysis.get("inflation", {})
        if inflation.get("inflation_level") == "high":
            weaknesses.append("High inflation pressures")
        
        return weaknesses
    
    def _predict_policy_responses(self, indicator_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Predict likely policy responses to economic conditions."""
        try:
            predictions = []
            
            inflation = indicator_analysis.get("inflation", {})
            if inflation.get("inflation_level") == "high":
                predictions.append("Central bank likely to maintain restrictive policy")
            elif inflation.get("inflation_level") in ["low", "deflationary"]:
                predictions.append("Central bank may consider easing measures")
            
            employment = indicator_analysis.get("employment", {})
            if employment.get("employment_health") == "weak":
                predictions.append("Fiscal stimulus measures may be considered")
            
            return {
                "monetary_policy_prediction": predictions,
                "fiscal_policy_likelihood": "medium",
                "timeline": "3_to_6_months"
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting policy responses: {e}")
            return {"error": str(e)}
    
    def _generate_currency_trading_implications(
        self, 
        currency_analysis: Dict[str, Any], 
        market_correlations: Dict[str, Any], 
        ai_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate currency trading implications."""
        try:
            trend = currency_analysis.get("trend", "stable")
            volatility = currency_analysis.get("volatility", 0)
            
            implications = []
            
            if trend == "strengthening":
                implications.append("Currency strength may pressure export-dependent stocks")
            elif trend == "weakening":
                implications.append("Currency weakness may benefit exporters")
            
            if volatility > 5:
                implications.append("High currency volatility increases hedging costs")
            
            return {
                "trading_implications": implications,
                "volatility_assessment": "high" if volatility > 5 else "moderate",
                "hedging_urgency": "high" if volatility > 10 else "medium"
            }
            
        except Exception as e:
            self.logger.error(f"Error generating currency trading implications: {e}")
            return {"error": str(e)}
    
    def _generate_hedging_recommendations(
        self, 
        currency_analysis: Dict[str, Any], 
        market_correlations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate currency hedging recommendations."""
        try:
            volatility = currency_analysis.get("volatility", 0)
            
            if volatility > 10:
                hedge_ratio = 0.8
                strategy = "aggressive_hedging"
            elif volatility > 5:
                hedge_ratio = 0.5
                strategy = "moderate_hedging"
            else:
                hedge_ratio = 0.2
                strategy = "minimal_hedging"
            
            return {
                "recommended_hedge_ratio": hedge_ratio,
                "hedging_strategy": strategy,
                "instruments": ["currency_forwards", "options"] if volatility > 5 else ["forwards"],
                "review_frequency": "monthly" if volatility > 5 else "quarterly"
            }
            
        except Exception as e:
            self.logger.error(f"Error generating hedging recommendations: {e}")
            return {"error": str(e)}
    
    def _generate_macro_outlook(
        self, 
        components: Dict[str, Any], 
        ai_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive macro outlook."""
        try:
            # Aggregate insights from all components
            outlook_factors = []
            
            monetary = components.get("monetary_policy", {})
            if monetary and "policy_implications" in monetary:
                policy_stance = monetary["policy_implications"].get("policy_stance", "neutral")
                outlook_factors.append(f"Monetary policy: {policy_stance}")
            
            economic = components.get("economic_indicators", {})
            if economic and "economic_outlook" in economic:
                econ_outlook = economic["economic_outlook"].get("outlook", "mixed")
                outlook_factors.append(f"Economic indicators: {econ_outlook}")
            
            # Overall assessment
            positive_factors = len([f for f in outlook_factors if "positive" in f or "strong" in f])
            negative_factors = len([f for f in outlook_factors if "negative" in f or "weak" in f])
            
            if positive_factors > negative_factors:
                overall_outlook = "positive"
            elif negative_factors > positive_factors:
                overall_outlook = "negative"
            else:
                overall_outlook = "mixed"
            
            return {
                "overall_outlook": overall_outlook,
                "key_factors": outlook_factors,
                "confidence_level": "medium",
                "timeline": "6_to_12_months",
                "major_risks": self._identify_macro_risks(components),
                "major_opportunities": self._identify_macro_opportunities(components)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating macro outlook: {e}")
            return {"error": str(e)}
    
    def _identify_macro_risks(self, components: Dict[str, Any]) -> List[str]:
        """Identify major macroeconomic risks."""
        risks = []
        
        monetary = components.get("monetary_policy", {})
        if monetary:
            rate_analysis = monetary.get("rate_analysis", {})
            if rate_analysis.get("short_term_trend") == "tightening":
                risks.append("Rising interest rates pressuring growth")
        
        economic = components.get("economic_indicators", {})
        if economic:
            weaknesses = economic.get("economic_outlook", {}).get("key_weaknesses", [])
            risks.extend(weaknesses)
        
        currency = components.get("currency_impact", {})
        if currency:
            volatility = currency.get("currency_analysis", {}).get("volatility", 0)
            if volatility > 10:
                risks.append("High currency volatility")
        
        return risks
    
    def _identify_macro_opportunities(self, components: Dict[str, Any]) -> List[str]:
        """Identify major macroeconomic opportunities."""
        opportunities = []
        
        monetary = components.get("monetary_policy", {})
        if monetary:
            rate_analysis = monetary.get("rate_analysis", {})
            if rate_analysis.get("rate_level") == "accommodative":
                opportunities.append("Supportive monetary policy environment")
        
        economic = components.get("economic_indicators", {})
        if economic:
            strengths = economic.get("economic_outlook", {}).get("key_strengths", [])
            opportunities.extend(strengths)
        
        return opportunities
    
    def _generate_strategic_recommendations(
        self, 
        macro_outlook: Dict[str, Any], 
        components: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate strategic investment recommendations."""
        try:
            overall_outlook = macro_outlook.get("overall_outlook", "mixed")
            
            if overall_outlook == "positive":
                strategy = "growth_oriented"
                allocation = {"equities": 0.7, "bonds": 0.2, "cash": 0.1}
            elif overall_outlook == "negative":
                strategy = "defensive"
                allocation = {"equities": 0.4, "bonds": 0.4, "cash": 0.2}
            else:
                strategy = "balanced"
                allocation = {"equities": 0.6, "bonds": 0.3, "cash": 0.1}
            
            return {
                "strategic_approach": strategy,
                "recommended_allocation": allocation,
                "rebalancing_frequency": "quarterly",
                "key_themes": macro_outlook.get("key_factors", []),
                "monitoring_priorities": ["interest_rates", "employment", "inflation"]
            }
            
        except Exception as e:
            self.logger.error(f"Error generating strategic recommendations: {e}")
            return {"error": str(e)}