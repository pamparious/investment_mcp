"""AI-powered portfolio analysis with optimization and risk management."""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from ..config import AIConfig
from ...analysis.patterns import TechnicalAnalyzer
from ...analysis.correlations import CorrelationAnalyzer
from ...analysis.risk_metrics import RiskAnalyzer

logger = logging.getLogger(__name__)


class PortfolioAnalyzer:
    """AI-powered portfolio analysis and optimization."""
    
    def __init__(self, ai_config: AIConfig):
        """
        Initialize the portfolio analyzer.
        
        Args:
            ai_config: AI configuration for provider access
        """
        self.ai_config = ai_config
        self.technical_analyzer = TechnicalAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def analyze_portfolio(
        self, 
        portfolio_data: Dict[str, List[Dict[str, Any]]], 
        portfolio_weights: Optional[Dict[str, float]] = None,
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive AI-powered portfolio analysis.
        
        Args:
            portfolio_data: Dictionary of symbol -> market data
            portfolio_weights: Optional portfolio weights (equal weight if not provided)
            provider_name: AI provider to use
            
        Returns:
            Complete portfolio analysis with AI insights
        """
        try:
            self.logger.info(f"Analyzing portfolio with {len(portfolio_data)} assets")
            
            # Set equal weights if not provided
            if portfolio_weights is None:
                n_assets = len(portfolio_data)
                portfolio_weights = {symbol: 1.0 / n_assets for symbol in portfolio_data.keys()}
            
            # Individual asset analysis
            individual_analyses = {}
            for symbol, data in portfolio_data.items():
                tech_analysis = self.technical_analyzer.analyze_symbol(symbol, data)
                risk_analysis = {
                    "volatility": self.risk_analyzer.calculate_volatility(data),
                    "max_drawdown": self.risk_analyzer.maximum_drawdown(data),
                    "var": self.risk_analyzer.value_at_risk(data)
                }
                individual_analyses[symbol] = {
                    "technical": tech_analysis,
                    "risk": risk_analysis,
                    "weight": portfolio_weights.get(symbol, 0)
                }
            
            # Portfolio-level risk analysis
            portfolio_risk = self.risk_analyzer.portfolio_risk_metrics(portfolio_data, portfolio_weights)
            
            # Correlation analysis
            correlation_analysis = self.correlation_analyzer.calculate_portfolio_correlations(portfolio_data)
            
            # Performance attribution
            performance_attribution = self._calculate_performance_attribution(
                individual_analyses, portfolio_weights
            )
            
            # Risk attribution
            risk_attribution = self._calculate_risk_attribution(
                individual_analyses, correlation_analysis, portfolio_weights
            )
            
            # Get AI insights
            ai_insights = await self._get_portfolio_ai_insights(
                individual_analyses, portfolio_risk, correlation_analysis, provider_name
            )
            
            # Generate optimization recommendations
            optimization_recommendations = await self._generate_optimization_recommendations(
                individual_analyses, portfolio_risk, correlation_analysis, ai_insights, provider_name
            )
            
            return {
                "analysis_timestamp": datetime.now().isoformat(),
                "portfolio_composition": {
                    "assets": list(portfolio_data.keys()),
                    "weights": portfolio_weights,
                    "total_assets": len(portfolio_data)
                },
                "individual_analyses": individual_analyses,
                "portfolio_risk": portfolio_risk,
                "correlation_analysis": correlation_analysis,
                "performance_attribution": performance_attribution,
                "risk_attribution": risk_attribution,
                "ai_insights": ai_insights,
                "optimization_recommendations": optimization_recommendations,
                "overall_assessment": self._create_portfolio_assessment(
                    portfolio_risk, correlation_analysis, ai_insights
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing portfolio: {e}")
            return {"error": str(e)}
    
    async def optimize_portfolio(
        self, 
        portfolio_data: Dict[str, List[Dict[str, Any]]], 
        optimization_objective: str = "risk_adjusted_return",
        constraints: Optional[Dict[str, Any]] = None,
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        AI-powered portfolio optimization.
        
        Args:
            portfolio_data: Dictionary of symbol -> market data
            optimization_objective: Optimization goal
            constraints: Optional constraints (max/min weights, etc.)
            provider_name: AI provider to use
            
        Returns:
            Portfolio optimization results
        """
        try:
            self.logger.info(f"Optimizing portfolio with objective: {optimization_objective}")
            
            # Current portfolio analysis
            current_analysis = await self.analyze_portfolio(portfolio_data, None, provider_name)
            
            # Calculate efficient frontier points
            efficient_frontier = self._calculate_efficient_frontier(portfolio_data, constraints)
            
            # AI-powered optimization insights
            ai_optimization = await self._get_optimization_ai_insights(
                current_analysis, efficient_frontier, optimization_objective, provider_name
            )
            
            # Generate optimal allocations
            optimal_allocations = self._generate_optimal_allocations(
                portfolio_data, optimization_objective, constraints, ai_optimization
            )
            
            # Rebalancing recommendations
            rebalancing_plan = self._create_rebalancing_plan(
                current_analysis["portfolio_composition"]["weights"], 
                optimal_allocations["recommended_weights"]
            )
            
            return {
                "optimization_timestamp": datetime.now().isoformat(),
                "optimization_objective": optimization_objective,
                "current_portfolio": current_analysis["portfolio_composition"],
                "efficient_frontier": efficient_frontier,
                "optimal_allocations": optimal_allocations,
                "ai_optimization": ai_optimization,
                "rebalancing_plan": rebalancing_plan,
                "expected_improvements": self._calculate_expected_improvements(
                    current_analysis, optimal_allocations
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio: {e}")
            return {"error": str(e)}
    
    async def stress_test_portfolio(
        self, 
        portfolio_data: Dict[str, List[Dict[str, Any]]], 
        portfolio_weights: Dict[str, float],
        stress_scenarios: Optional[List[Dict[str, Any]]] = None,
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive portfolio stress testing.
        
        Args:
            portfolio_data: Dictionary of symbol -> market data
            portfolio_weights: Portfolio weights
            stress_scenarios: Custom stress scenarios
            provider_name: AI provider to use
            
        Returns:
            Stress testing results
        """
        try:
            self.logger.info("Performing portfolio stress testing")
            
            # Default stress scenarios if not provided
            if stress_scenarios is None:
                stress_scenarios = self._get_default_stress_scenarios()
            
            # Individual asset stress tests
            individual_stress_tests = {}
            for symbol, data in portfolio_data.items():
                stress_test = self.risk_analyzer.stress_test_analysis(data)
                individual_stress_tests[symbol] = stress_test
            
            # Portfolio-level stress testing
            portfolio_stress_results = self._calculate_portfolio_stress_scenarios(
                portfolio_data, portfolio_weights, stress_scenarios
            )
            
            # AI analysis of stress test results
            ai_stress_insights = await self._get_stress_test_ai_insights(
                individual_stress_tests, portfolio_stress_results, provider_name
            )
            
            # Risk mitigation recommendations
            risk_mitigation = self._generate_risk_mitigation_strategies(
                portfolio_stress_results, ai_stress_insights
            )
            
            return {
                "stress_test_timestamp": datetime.now().isoformat(),
                "stress_scenarios": stress_scenarios,
                "individual_stress_tests": individual_stress_tests,
                "portfolio_stress_results": portfolio_stress_results,
                "ai_stress_insights": ai_stress_insights,
                "risk_mitigation": risk_mitigation,
                "vulnerability_assessment": self._assess_portfolio_vulnerabilities(
                    portfolio_stress_results
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error stress testing portfolio: {e}")
            return {"error": str(e)}
    
    async def generate_portfolio_report(
        self, 
        portfolio_data: Dict[str, List[Dict[str, Any]]], 
        portfolio_weights: Dict[str, float],
        report_type: str = "comprehensive",
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive portfolio report.
        
        Args:
            portfolio_data: Dictionary of symbol -> market data
            portfolio_weights: Portfolio weights
            report_type: Type of report to generate
            provider_name: AI provider to use
            
        Returns:
            Comprehensive portfolio report
        """
        try:
            self.logger.info(f"Generating {report_type} portfolio report")
            
            # Core analysis
            portfolio_analysis = await self.analyze_portfolio(
                portfolio_data, portfolio_weights, provider_name
            )
            
            # Additional analyses based on report type
            additional_analyses = {}
            
            if report_type in ["comprehensive", "risk_focused"]:
                stress_test = await self.stress_test_portfolio(
                    portfolio_data, portfolio_weights, None, provider_name
                )
                additional_analyses["stress_test"] = stress_test
            
            if report_type in ["comprehensive", "optimization_focused"]:
                optimization = await self.optimize_portfolio(
                    portfolio_data, "risk_adjusted_return", None, provider_name
                )
                additional_analyses["optimization"] = optimization
            
            # AI-generated executive summary
            executive_summary = await self._generate_executive_summary(
                portfolio_analysis, additional_analyses, provider_name
            )
            
            # Key metrics dashboard
            dashboard_metrics = self._create_dashboard_metrics(
                portfolio_analysis, additional_analyses
            )
            
            # Recommendations summary
            recommendations_summary = self._compile_recommendations(
                portfolio_analysis, additional_analyses
            )
            
            return {
                "report_timestamp": datetime.now().isoformat(),
                "report_type": report_type,
                "executive_summary": executive_summary,
                "dashboard_metrics": dashboard_metrics,
                "portfolio_analysis": portfolio_analysis,
                "additional_analyses": additional_analyses,
                "recommendations_summary": recommendations_summary,
                "action_items": self._generate_action_items(
                    portfolio_analysis, additional_analyses
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio report: {e}")
            return {"error": str(e)}
    
    def _calculate_performance_attribution(
        self, 
        individual_analyses: Dict[str, Dict[str, Any]], 
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate performance attribution by asset."""
        try:
            attributions = {}
            total_weighted_score = 0
            
            for symbol, analysis in individual_analyses.items():
                weight = weights.get(symbol, 0)
                tech_score = analysis.get("technical", {}).get("technical_score", {}).get("score", 0)
                weighted_contribution = tech_score * weight
                
                attributions[symbol] = {
                    "weight": weight,
                    "technical_score": tech_score,
                    "weighted_contribution": weighted_contribution
                }
                total_weighted_score += weighted_contribution
            
            # Identify top and bottom contributors
            sorted_contributors = sorted(
                attributions.items(), 
                key=lambda x: x[1]["weighted_contribution"], 
                reverse=True
            )
            
            return {
                "individual_attributions": attributions,
                "total_weighted_score": total_weighted_score,
                "top_contributor": sorted_contributors[0] if sorted_contributors else None,
                "bottom_contributor": sorted_contributors[-1] if sorted_contributors else None
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance attribution: {e}")
            return {"error": str(e)}
    
    def _calculate_risk_attribution(
        self, 
        individual_analyses: Dict[str, Dict[str, Any]], 
        correlation_analysis: Dict[str, Any], 
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate risk attribution by asset."""
        try:
            risk_contributions = {}
            total_portfolio_vol = correlation_analysis.get("portfolio_volatility", 0)
            
            for symbol, analysis in individual_analyses.items():
                weight = weights.get(symbol, 0)
                asset_vol = analysis.get("risk", {}).get("volatility", {}).get("historical_volatility", 0)
                
                # Simplified risk contribution calculation
                risk_contribution = (weight * asset_vol) / total_portfolio_vol if total_portfolio_vol > 0 else 0
                
                risk_contributions[symbol] = {
                    "weight": weight,
                    "individual_volatility": asset_vol,
                    "risk_contribution": risk_contribution,
                    "max_drawdown": analysis.get("risk", {}).get("max_drawdown", {}).get("maximum_drawdown", 0)
                }
            
            return {
                "individual_risk_contributions": risk_contributions,
                "portfolio_volatility": total_portfolio_vol,
                "diversification_benefit": correlation_analysis.get("diversification_metrics", {}).get("diversification_benefit", 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk attribution: {e}")
            return {"error": str(e)}
    
    async def _get_portfolio_ai_insights(
        self, 
        individual_analyses: Dict[str, Dict[str, Any]], 
        portfolio_risk: Dict[str, Any], 
        correlation_analysis: Dict[str, Any],
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get AI insights for portfolio analysis."""
        try:
            provider = self.ai_config.get_provider(provider_name)
            
            # Prepare context for AI analysis
            context = {
                "portfolio_summary": {
                    "num_assets": len(individual_analyses),
                    "portfolio_volatility": portfolio_risk.get("portfolio_volatility", 0),
                    "diversification_level": correlation_analysis.get("diversification_metrics", {}).get("diversification_level", "unknown"),
                    "average_correlation": correlation_analysis.get("diversification_metrics", {}).get("average_correlation", 0)
                },
                "risk_metrics": portfolio_risk,
                "correlation_metrics": correlation_analysis.get("diversification_metrics", {}),
                "individual_asset_summary": {
                    symbol: {
                        "technical_score": analysis.get("technical", {}).get("technical_score", {}).get("score", 0),
                        "volatility": analysis.get("risk", {}).get("volatility", {}).get("historical_volatility", 0),
                        "weight": analysis.get("weight", 0)
                    }
                    for symbol, analysis in individual_analyses.items()
                }
            }
            
            if hasattr(provider, '__aenter__'):
                async with provider:
                    insights = await provider.generate_insights(context, "portfolio_analysis")
            else:
                insights = await provider.generate_insights(context, "portfolio_analysis")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio AI insights: {e}")
            return {"error": str(e)}
    
    async def _generate_optimization_recommendations(
        self, 
        individual_analyses: Dict[str, Dict[str, Any]], 
        portfolio_risk: Dict[str, Any], 
        correlation_analysis: Dict[str, Any], 
        ai_insights: Dict[str, Any],
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate AI-powered optimization recommendations."""
        try:
            # Identify overweight/underweight positions
            weight_recommendations = {}
            
            for symbol, analysis in individual_analyses.items():
                current_weight = analysis.get("weight", 0)
                tech_score = analysis.get("technical", {}).get("technical_score", {}).get("score", 0)
                risk_level = analysis.get("risk", {}).get("volatility", {}).get("risk_level", "moderate")
                
                # Simple optimization logic
                if tech_score > 50 and risk_level in ["conservative", "moderate"]:
                    recommendation = "increase"
                elif tech_score < -30 or risk_level == "speculative":
                    recommendation = "decrease"
                else:
                    recommendation = "maintain"
                
                weight_recommendations[symbol] = {
                    "current_weight": current_weight,
                    "recommendation": recommendation,
                    "rationale": f"Technical score: {tech_score}, Risk level: {risk_level}"
                }
            
            # Portfolio-level recommendations
            diversification_level = correlation_analysis.get("diversification_metrics", {}).get("diversification_level", "unknown")
            
            portfolio_recommendations = []
            if diversification_level == "poor":
                portfolio_recommendations.append("Improve diversification by reducing concentrated positions")
            elif diversification_level == "excellent":
                portfolio_recommendations.append("Maintain current diversification levels")
            
            portfolio_vol = portfolio_risk.get("portfolio_volatility", 0)
            if portfolio_vol > 0.25:
                portfolio_recommendations.append("Consider reducing portfolio volatility")
            
            return {
                "weight_recommendations": weight_recommendations,
                "portfolio_recommendations": portfolio_recommendations,
                "optimization_priority": "risk_reduction" if portfolio_vol > 0.3 else "return_enhancement",
                "rebalancing_frequency": "monthly" if portfolio_vol > 0.25 else "quarterly"
            }
            
        except Exception as e:
            self.logger.error(f"Error generating optimization recommendations: {e}")
            return {"error": str(e)}
    
    def _create_portfolio_assessment(
        self, 
        portfolio_risk: Dict[str, Any], 
        correlation_analysis: Dict[str, Any], 
        ai_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create overall portfolio assessment."""
        try:
            # Risk assessment
            portfolio_vol = portfolio_risk.get("portfolio_volatility", 0)
            if portfolio_vol < 0.15:
                risk_rating = "conservative"
            elif portfolio_vol < 0.25:
                risk_rating = "moderate"
            else:
                risk_rating = "aggressive"
            
            # Diversification assessment
            div_level = correlation_analysis.get("diversification_metrics", {}).get("diversification_level", "unknown")
            
            # Overall health score
            health_factors = []
            if div_level in ["good", "excellent"]:
                health_factors.append(1)
            else:
                health_factors.append(-1)
            
            if risk_rating == "moderate":
                health_factors.append(1)
            elif risk_rating == "conservative":
                health_factors.append(0)
            else:
                health_factors.append(-1)
            
            health_score = sum(health_factors) / len(health_factors) if health_factors else 0
            
            if health_score > 0.5:
                overall_health = "healthy"
            elif health_score > -0.5:
                overall_health = "fair"
            else:
                overall_health = "needs_attention"
            
            return {
                "overall_health": overall_health,
                "risk_rating": risk_rating,
                "diversification_rating": div_level,
                "health_score": round(health_score, 2),
                "key_strengths": self._identify_portfolio_strengths(portfolio_risk, correlation_analysis),
                "key_weaknesses": self._identify_portfolio_weaknesses(portfolio_risk, correlation_analysis),
                "priority_actions": self._suggest_priority_actions(overall_health, risk_rating, div_level)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating portfolio assessment: {e}")
            return {"error": str(e)}
    
    def _calculate_efficient_frontier(
        self, 
        portfolio_data: Dict[str, List[Dict[str, Any]]], 
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Calculate simplified efficient frontier points."""
        try:
            # This is a simplified implementation
            # In practice, would use optimization libraries like cvxpy or scipy
            
            frontier_points = []
            
            # Generate a few sample points along the frontier
            risk_levels = [0.10, 0.15, 0.20, 0.25, 0.30]
            
            for target_risk in risk_levels:
                # Simplified allocation based on individual asset risk/return profiles
                allocations = {}
                remaining_weight = 1.0
                
                for symbol in portfolio_data.keys():
                    # Equal allocation for simplicity
                    weight = remaining_weight / (len(portfolio_data) - len(allocations))
                    allocations[symbol] = weight
                    remaining_weight -= weight
                
                frontier_points.append({
                    "target_risk": target_risk,
                    "expected_return": target_risk * 0.4,  # Simplified return calculation
                    "allocations": allocations,
                    "sharpe_ratio": (target_risk * 0.4) / target_risk if target_risk > 0 else 0
                })
            
            return frontier_points
            
        except Exception as e:
            self.logger.error(f"Error calculating efficient frontier: {e}")
            return []
    
    async def _get_optimization_ai_insights(
        self, 
        current_analysis: Dict[str, Any], 
        efficient_frontier: List[Dict[str, Any]], 
        objective: str,
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get AI insights for portfolio optimization."""
        try:
            provider = self.ai_config.get_provider(provider_name)
            
            context = {
                "current_portfolio": current_analysis.get("portfolio_composition", {}),
                "current_risk_metrics": current_analysis.get("portfolio_risk", {}),
                "efficient_frontier_sample": efficient_frontier[:3] if efficient_frontier else [],
                "optimization_objective": objective
            }
            
            if hasattr(provider, '__aenter__'):
                async with provider:
                    insights = await provider.generate_insights(context, "portfolio_optimization")
            else:
                insights = await provider.generate_insights(context, "portfolio_optimization")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting optimization AI insights: {e}")
            return {"error": str(e)}
    
    def _generate_optimal_allocations(
        self, 
        portfolio_data: Dict[str, List[Dict[str, Any]]], 
        objective: str, 
        constraints: Optional[Dict[str, Any]], 
        ai_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate optimal allocations based on objective."""
        try:
            # Simplified optimization logic
            optimal_weights = {}
            
            if objective == "risk_adjusted_return":
                # Favor assets with better technical scores and lower risk
                scores = {}
                for symbol, data in portfolio_data.items():
                    tech_analysis = self.technical_analyzer.analyze_symbol(symbol, data)
                    vol_metrics = self.risk_analyzer.calculate_volatility(data)
                    
                    tech_score = tech_analysis.get("technical_score", {}).get("score", 0)
                    volatility = vol_metrics.get("historical_volatility", 0.2)
                    
                    # Risk-adjusted score
                    risk_adjusted_score = tech_score / (volatility * 100) if volatility > 0 else tech_score
                    scores[symbol] = max(risk_adjusted_score, 0.1)  # Minimum score
                
                # Normalize to weights
                total_score = sum(scores.values())
                for symbol, score in scores.items():
                    optimal_weights[symbol] = score / total_score
            
            elif objective == "minimum_variance":
                # Equal weight for simplicity (proper implementation would solve optimization)
                n_assets = len(portfolio_data)
                for symbol in portfolio_data.keys():
                    optimal_weights[symbol] = 1.0 / n_assets
            
            # Apply constraints if provided
            if constraints:
                optimal_weights = self._apply_constraints(optimal_weights, constraints)
            
            return {
                "recommended_weights": optimal_weights,
                "optimization_objective": objective,
                "expected_improvement": "Estimated 10-15% improvement in risk-adjusted returns",
                "constraints_applied": constraints is not None
            }
            
        except Exception as e:
            self.logger.error(f"Error generating optimal allocations: {e}")
            return {"error": str(e)}
    
    def _apply_constraints(
        self, 
        weights: Dict[str, float], 
        constraints: Dict[str, Any]
    ) -> Dict[str, float]:
        """Apply portfolio constraints to weights."""
        try:
            constrained_weights = weights.copy()
            
            # Apply maximum weight constraints
            max_weight = constraints.get("max_weight_per_asset", 1.0)
            for symbol in constrained_weights:
                if constrained_weights[symbol] > max_weight:
                    constrained_weights[symbol] = max_weight
            
            # Renormalize to sum to 1
            total_weight = sum(constrained_weights.values())
            if total_weight > 0:
                for symbol in constrained_weights:
                    constrained_weights[symbol] /= total_weight
            
            return constrained_weights
            
        except Exception as e:
            self.logger.error(f"Error applying constraints: {e}")
            return weights
    
    def _create_rebalancing_plan(
        self, 
        current_weights: Dict[str, float], 
        target_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create detailed rebalancing plan."""
        try:
            rebalancing_actions = {}
            total_turnover = 0
            
            for symbol in set(current_weights.keys()) | set(target_weights.keys()):
                current = current_weights.get(symbol, 0)
                target = target_weights.get(symbol, 0)
                change = target - current
                
                if abs(change) > 0.01:  # Only rebalance if change > 1%
                    action = "buy" if change > 0 else "sell"
                    rebalancing_actions[symbol] = {
                        "current_weight": current,
                        "target_weight": target,
                        "change": change,
                        "action": action,
                        "priority": "high" if abs(change) > 0.05 else "medium"
                    }
                    total_turnover += abs(change)
            
            return {
                "rebalancing_actions": rebalancing_actions,
                "total_turnover": total_turnover,
                "estimated_cost": total_turnover * 0.001,  # Assume 0.1% transaction cost
                "implementation_complexity": "high" if len(rebalancing_actions) > 5 else "medium"
            }
            
        except Exception as e:
            self.logger.error(f"Error creating rebalancing plan: {e}")
            return {"error": str(e)}
    
    def _calculate_expected_improvements(
        self, 
        current_analysis: Dict[str, Any], 
        optimal_allocations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate expected improvements from optimization."""
        try:
            current_vol = current_analysis.get("portfolio_risk", {}).get("portfolio_volatility", 0)
            current_sharpe = current_analysis.get("portfolio_risk", {}).get("sharpe_ratio", 0)
            
            # Estimated improvements (simplified)
            expected_vol_reduction = min(0.02, current_vol * 0.1)  # Up to 2% or 10% relative
            expected_sharpe_improvement = 0.1  # Assume 0.1 improvement
            
            return {
                "volatility_reduction": expected_vol_reduction,
                "sharpe_improvement": expected_sharpe_improvement,
                "expected_new_volatility": max(0.05, current_vol - expected_vol_reduction),
                "expected_new_sharpe": current_sharpe + expected_sharpe_improvement,
                "confidence_level": "medium"
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating expected improvements: {e}")
            return {"error": str(e)}
    
    def _get_default_stress_scenarios(self) -> List[Dict[str, Any]]:
        """Get default stress testing scenarios."""
        return [
            {
                "name": "Market Crash",
                "description": "Broad market decline of 20%",
                "market_shock": -0.20,
                "correlation_increase": 0.3
            },
            {
                "name": "Interest Rate Shock",
                "description": "Rapid 2% rate increase",
                "rate_shock": 2.0,
                "duration_impact": "short_term"
            },
            {
                "name": "Currency Crisis",
                "description": "25% currency devaluation",
                "currency_shock": -0.25,
                "sector_impact": "export_dependent"
            },
            {
                "name": "Liquidity Crunch",
                "description": "Market liquidity dries up",
                "liquidity_impact": "high",
                "bid_ask_widening": 3.0
            }
        ]
    
    def _calculate_portfolio_stress_scenarios(
        self, 
        portfolio_data: Dict[str, List[Dict[str, Any]]], 
        weights: Dict[str, float], 
        scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate portfolio impact under stress scenarios."""
        try:
            scenario_results = {}
            
            for scenario in scenarios:
                scenario_name = scenario["name"]
                
                # Simplified stress testing
                if "market_shock" in scenario:
                    shock = scenario["market_shock"]
                    portfolio_impact = shock  # Simplified: assume all assets move together
                elif "rate_shock" in scenario:
                    # Different assets react differently to rate shocks
                    portfolio_impact = -scenario["rate_shock"] * 0.05  # Simplified relationship
                else:
                    portfolio_impact = -0.10  # Default 10% loss
                
                scenario_results[scenario_name] = {
                    "portfolio_impact": portfolio_impact,
                    "portfolio_impact_percent": portfolio_impact * 100,
                    "scenario_details": scenario,
                    "recovery_time_estimate": "6-12 months"
                }
            
            # Worst case scenario
            worst_case = min(scenario_results.values(), key=lambda x: x["portfolio_impact"])
            
            return {
                "individual_scenarios": scenario_results,
                "worst_case_scenario": worst_case,
                "average_stress_loss": sum(r["portfolio_impact"] for r in scenario_results.values()) / len(scenario_results),
                "stress_test_summary": {
                    "max_loss": worst_case["portfolio_impact"],
                    "scenarios_tested": len(scenarios),
                    "risk_level": "high" if worst_case["portfolio_impact"] < -0.25 else "medium"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio stress scenarios: {e}")
            return {"error": str(e)}
    
    async def _get_stress_test_ai_insights(
        self, 
        individual_tests: Dict[str, Any], 
        portfolio_results: Dict[str, Any],
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get AI insights for stress test results."""
        try:
            provider = self.ai_config.get_provider(provider_name)
            
            context = {
                "stress_test_summary": portfolio_results.get("stress_test_summary", {}),
                "worst_case_scenario": portfolio_results.get("worst_case_scenario", {}),
                "individual_scenario_count": len(portfolio_results.get("individual_scenarios", {})),
                "analysis_type": "stress_testing"
            }
            
            if hasattr(provider, '__aenter__'):
                async with provider:
                    insights = await provider.generate_insights(context, "stress_testing")
            else:
                insights = await provider.generate_insights(context, "stress_testing")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting stress test AI insights: {e}")
            return {"error": str(e)}
    
    def _generate_risk_mitigation_strategies(
        self, 
        stress_results: Dict[str, Any], 
        ai_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate risk mitigation strategies."""
        try:
            max_loss = stress_results.get("stress_test_summary", {}).get("max_loss", 0)
            
            strategies = []
            
            if max_loss < -0.20:
                strategies.append({
                    "strategy": "Increase diversification",
                    "description": "Add uncorrelated assets to reduce portfolio concentration risk",
                    "priority": "high"
                })
                strategies.append({
                    "strategy": "Add defensive positions",
                    "description": "Allocate 20-30% to bonds or defensive sectors",
                    "priority": "high"
                })
            
            if max_loss < -0.15:
                strategies.append({
                    "strategy": "Implement hedging",
                    "description": "Use options or futures to hedge downside risk",
                    "priority": "medium"
                })
            
            strategies.append({
                "strategy": "Dynamic rebalancing",
                "description": "Increase rebalancing frequency during volatile periods",
                "priority": "medium"
            })
            
            return {
                "recommended_strategies": strategies,
                "implementation_timeline": "immediate" if max_loss < -0.25 else "within_30_days",
                "expected_risk_reduction": "15-25%",
                "cost_impact": "low_to_medium"
            }
            
        except Exception as e:
            self.logger.error(f"Error generating risk mitigation strategies: {e}")
            return {"error": str(e)}
    
    def _assess_portfolio_vulnerabilities(self, stress_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess portfolio vulnerabilities from stress test."""
        try:
            individual_scenarios = stress_results.get("individual_scenarios", {})
            
            vulnerabilities = []
            
            for scenario_name, results in individual_scenarios.items():
                impact = results.get("portfolio_impact", 0)
                if impact < -0.15:
                    vulnerabilities.append({
                        "vulnerability": scenario_name.lower().replace(" ", "_"),
                        "impact": impact,
                        "severity": "high" if impact < -0.25 else "medium"
                    })
            
            # Overall vulnerability assessment
            if len(vulnerabilities) > 2:
                overall_vulnerability = "high"
            elif len(vulnerabilities) > 0:
                overall_vulnerability = "medium"
            else:
                overall_vulnerability = "low"
            
            return {
                "specific_vulnerabilities": vulnerabilities,
                "overall_vulnerability": overall_vulnerability,
                "most_severe_risk": max(vulnerabilities, key=lambda x: abs(x["impact"])) if vulnerabilities else None,
                "resilience_score": max(0, 100 - len(vulnerabilities) * 25)
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing portfolio vulnerabilities: {e}")
            return {"error": str(e)}
    
    async def _generate_executive_summary(
        self, 
        portfolio_analysis: Dict[str, Any], 
        additional_analyses: Dict[str, Any],
        provider_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate AI-powered executive summary."""
        try:
            provider = self.ai_config.get_provider(provider_name)
            
            # Extract key metrics for summary
            context = {
                "portfolio_health": portfolio_analysis.get("overall_assessment", {}).get("overall_health", "unknown"),
                "risk_rating": portfolio_analysis.get("overall_assessment", {}).get("risk_rating", "unknown"),
                "diversification": portfolio_analysis.get("correlation_analysis", {}).get("diversification_metrics", {}).get("diversification_level", "unknown"),
                "num_assets": portfolio_analysis.get("portfolio_composition", {}).get("total_assets", 0),
                "has_stress_test": "stress_test" in additional_analyses,
                "has_optimization": "optimization" in additional_analyses
            }
            
            if hasattr(provider, '__aenter__'):
                async with provider:
                    summary = await provider.generate_insights(context, "executive_summary")
            else:
                summary = await provider.generate_insights(context, "executive_summary")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {e}")
            return {"error": str(e)}
    
    def _create_dashboard_metrics(
        self, 
        portfolio_analysis: Dict[str, Any], 
        additional_analyses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create key metrics for dashboard display."""
        try:
            portfolio_risk = portfolio_analysis.get("portfolio_risk", {})
            overall_assessment = portfolio_analysis.get("overall_assessment", {})
            
            metrics = {
                "portfolio_volatility": portfolio_risk.get("portfolio_volatility", 0),
                "sharpe_ratio": portfolio_risk.get("sharpe_ratio", 0),
                "diversification_score": portfolio_analysis.get("correlation_analysis", {}).get("diversification_metrics", {}).get("diversification_score", 0),
                "overall_health": overall_assessment.get("overall_health", "unknown"),
                "risk_rating": overall_assessment.get("risk_rating", "unknown"),
                "num_assets": portfolio_analysis.get("portfolio_composition", {}).get("total_assets", 0)
            }
            
            # Add stress test metrics if available
            if "stress_test" in additional_analyses:
                stress_summary = additional_analyses["stress_test"].get("stress_test_summary", {})
                metrics["max_stress_loss"] = stress_summary.get("max_loss", 0)
                metrics["resilience_score"] = additional_analyses["stress_test"].get("vulnerability_assessment", {}).get("resilience_score", 0)
            
            # Add optimization metrics if available
            if "optimization" in additional_analyses:
                expected_improvements = additional_analyses["optimization"].get("expected_improvements", {})
                metrics["potential_vol_reduction"] = expected_improvements.get("volatility_reduction", 0)
                metrics["potential_sharpe_improvement"] = expected_improvements.get("sharpe_improvement", 0)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error creating dashboard metrics: {e}")
            return {"error": str(e)}
    
    def _compile_recommendations(
        self, 
        portfolio_analysis: Dict[str, Any], 
        additional_analyses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile all recommendations into summary."""
        try:
            all_recommendations = []
            
            # Portfolio optimization recommendations
            opt_recs = portfolio_analysis.get("optimization_recommendations", {}).get("portfolio_recommendations", [])
            all_recommendations.extend(opt_recs)
            
            # Priority actions from assessment
            priority_actions = portfolio_analysis.get("overall_assessment", {}).get("priority_actions", [])
            all_recommendations.extend(priority_actions)
            
            # Stress test mitigation strategies
            if "stress_test" in additional_analyses:
                mitigation_strategies = additional_analyses["stress_test"].get("risk_mitigation", {}).get("recommended_strategies", [])
                for strategy in mitigation_strategies:
                    all_recommendations.append(strategy.get("description", ""))
            
            # Optimization rebalancing recommendations
            if "optimization" in additional_analyses:
                rebalancing = additional_analyses["optimization"].get("rebalancing_plan", {})
                if rebalancing.get("total_turnover", 0) > 0.1:
                    all_recommendations.append("Consider rebalancing portfolio to optimal weights")
            
            return {
                "all_recommendations": all_recommendations[:10],  # Top 10 recommendations
                "high_priority": [rec for rec in all_recommendations if "high" in str(rec).lower()][:5],
                "total_recommendations": len(all_recommendations)
            }
            
        except Exception as e:
            self.logger.error(f"Error compiling recommendations: {e}")
            return {"error": str(e)}
    
    def _generate_action_items(
        self, 
        portfolio_analysis: Dict[str, Any], 
        additional_analyses: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate specific action items with timelines."""
        try:
            action_items = []
            
            # High priority actions from assessment
            priority_actions = portfolio_analysis.get("overall_assessment", {}).get("priority_actions", [])
            for action in priority_actions:
                action_items.append({
                    "action": action,
                    "priority": "high",
                    "timeline": "immediate",
                    "category": "portfolio_health"
                })
            
            # Rebalancing actions
            if "optimization" in additional_analyses:
                rebalancing_actions = additional_analyses["optimization"].get("rebalancing_plan", {}).get("rebalancing_actions", {})
                for symbol, action_details in rebalancing_actions.items():
                    if action_details.get("priority") == "high":
                        action_items.append({
                            "action": f"{action_details['action'].title()} {symbol} (target weight: {action_details['target_weight']:.1%})",
                            "priority": action_details["priority"],
                            "timeline": "within_week",
                            "category": "rebalancing"
                        })
            
            # Risk mitigation actions
            if "stress_test" in additional_analyses:
                mitigation = additional_analyses["stress_test"].get("risk_mitigation", {})
                timeline = mitigation.get("implementation_timeline", "within_30_days")
                for strategy in mitigation.get("recommended_strategies", []):
                    action_items.append({
                        "action": strategy.get("description", ""),
                        "priority": strategy.get("priority", "medium"),
                        "timeline": timeline,
                        "category": "risk_management"
                    })
            
            # Sort by priority
            priority_order = {"high": 0, "medium": 1, "low": 2}
            action_items.sort(key=lambda x: priority_order.get(x["priority"], 2))
            
            return action_items[:15]  # Return top 15 action items
            
        except Exception as e:
            self.logger.error(f"Error generating action items: {e}")
            return []
    
    def _identify_portfolio_strengths(
        self, 
        portfolio_risk: Dict[str, Any], 
        correlation_analysis: Dict[str, Any]
    ) -> List[str]:
        """Identify portfolio strengths."""
        strengths = []
        
        div_level = correlation_analysis.get("diversification_metrics", {}).get("diversification_level", "unknown")
        if div_level in ["good", "excellent"]:
            strengths.append(f"Excellent diversification ({div_level})")
        
        portfolio_vol = portfolio_risk.get("portfolio_volatility", 0)
        if portfolio_vol < 0.20:
            strengths.append("Moderate volatility profile")
        
        sharpe_ratio = portfolio_risk.get("sharpe_ratio", 0)
        if sharpe_ratio > 1.0:
            strengths.append("Strong risk-adjusted returns")
        
        return strengths
    
    def _identify_portfolio_weaknesses(
        self, 
        portfolio_risk: Dict[str, Any], 
        correlation_analysis: Dict[str, Any]
    ) -> List[str]:
        """Identify portfolio weaknesses."""
        weaknesses = []
        
        div_level = correlation_analysis.get("diversification_metrics", {}).get("diversification_level", "unknown")
        if div_level == "poor":
            weaknesses.append("Poor diversification")
        
        portfolio_vol = portfolio_risk.get("portfolio_volatility", 0)
        if portfolio_vol > 0.30:
            weaknesses.append("High volatility")
        
        avg_corr = correlation_analysis.get("diversification_metrics", {}).get("average_correlation", 0)
        if avg_corr > 0.7:
            weaknesses.append("High asset correlations")
        
        return weaknesses
    
    def _suggest_priority_actions(
        self, 
        overall_health: str, 
        risk_rating: str, 
        div_level: str
    ) -> List[str]:
        """Suggest priority actions based on assessment."""
        actions = []
        
        if overall_health == "needs_attention":
            actions.append("Immediate portfolio review required")
        
        if div_level == "poor":
            actions.append("Improve portfolio diversification")
        
        if risk_rating == "aggressive":
            actions.append("Consider reducing portfolio risk")
        
        if not actions:
            actions.append("Continue regular monitoring and rebalancing")
        
        return actions