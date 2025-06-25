"""Portfolio analysis service for Investment MCP API."""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ...models.requests import PortfolioConstraints, RiskProfile
from ...common.exceptions import (
    PortfolioOptimizationException,
    InsufficientDataException,
    AIServiceException
)


logger = logging.getLogger(__name__)


class PortfolioAnalysisService:
    """Service for comprehensive portfolio analysis."""
    
    def __init__(self):
        """Initialize portfolio analysis service."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def analyze_portfolio(
        self,
        risk_profile: RiskProfile,
        investment_amount: float,
        investment_horizon_years: int,
        current_allocation: Optional[Dict[str, float]] = None,
        constraints: Optional[PortfolioConstraints] = None,
        include_stress_test: bool = True,
        include_historical_analysis: bool = True,
        user_tier: str = "standard"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive portfolio analysis.
        
        Args:
            risk_profile: Investor risk tolerance
            investment_amount: Investment amount in SEK
            investment_horizon_years: Investment time horizon
            current_allocation: Current portfolio allocation
            constraints: Portfolio constraints
            include_stress_test: Whether to include stress testing
            include_historical_analysis: Whether to include historical analysis
            user_tier: User subscription tier
            
        Returns:
            Complete portfolio analysis results
        """
        
        self.logger.info(f"Starting portfolio analysis for {risk_profile} investor")
        
        try:
            # Step 1: Collect and validate data
            data_quality = await self._collect_and_validate_data()
            
            # Step 2: Generate optimal allocation
            allocation_result = await self._generate_optimal_allocation(
                risk_profile, investment_horizon_years, constraints, user_tier
            )
            
            # Step 3: Calculate expected metrics
            expected_metrics = await self._calculate_portfolio_metrics(
                allocation_result["allocation"]
            )
            
            # Step 4: Historical analysis (if requested)
            historical_analysis = None
            if include_historical_analysis:
                historical_analysis = await self._perform_historical_analysis(
                    allocation_result["allocation"]
                )
            
            # Step 5: Stress testing (if requested)
            stress_test_results = None
            if include_stress_test:
                stress_test_results = await self._perform_stress_testing(
                    allocation_result["allocation"]
                )
            
            # Step 6: Swedish economic context
            swedish_context = await self._get_swedish_economic_context()
            
            # Step 7: Market regime analysis
            regime_analysis = await self._analyze_market_regimes(
                allocation_result["allocation"]
            )
            
            # Step 8: Generate AI reasoning
            ai_reasoning = await self._generate_ai_reasoning(
                allocation_result, expected_metrics, swedish_context
            )
            
            # Step 9: Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                data_quality, allocation_result, expected_metrics
            )
            
            # Compile final results
            results = {
                "allocation": allocation_result["allocation"],
                "expected_metrics": expected_metrics,
                "historical_analysis": historical_analysis,
                "stress_test": stress_test_results,
                "regime_analysis": regime_analysis,
                "swedish_economic_context": swedish_context,
                "ai_reasoning": ai_reasoning,
                "confidence_score": confidence_score,
                "data_quality_score": data_quality["overall_score"],
                "optimization_details": allocation_result.get("details", {})
            }
            
            self.logger.info(f"Portfolio analysis completed with confidence {confidence_score:.2f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Portfolio analysis failed: {e}")
            raise PortfolioOptimizationException(
                detail=f"Portfolio analysis failed: {str(e)}",
                optimization_method="ai_guided"
            )
    
    async def _collect_and_validate_data(self) -> Dict[str, Any]:
        """Collect and validate required data for analysis."""
        
        self.logger.info("Collecting and validating data")
        
        # Simulate data collection and quality assessment
        # In real implementation, this would:
        # 1. Fetch historical fund data
        # 2. Validate data completeness and quality
        # 3. Check for missing periods or anomalies
        
        data_quality = {
            "funds_available": 12,
            "avg_data_coverage": 0.95,
            "historical_years": 20,
            "missing_data_periods": 2,
            "data_freshness_hours": 2,
            "overall_score": 0.88
        }
        
        if data_quality["overall_score"] < 0.7:
            raise InsufficientDataException(
                detail="Data quality insufficient for reliable analysis",
                available_period=f"{data_quality['historical_years']} years",
                required_period="15+ years"
            )
        
        return data_quality
    
    async def _generate_optimal_allocation(
        self,
        risk_profile: RiskProfile,
        investment_horizon: int,
        constraints: Optional[PortfolioConstraints],
        user_tier: str
    ) -> Dict[str, Any]:
        """Generate optimal portfolio allocation."""
        
        self.logger.info(f"Generating optimal allocation for {risk_profile} profile")
        
        # Risk profile mappings
        risk_mappings = {
            RiskProfile.CONSERVATIVE: {
                "max_equity": 0.6,
                "min_bonds": 0.3,
                "max_alternatives": 0.1,
                "target_volatility": 0.12
            },
            RiskProfile.BALANCED: {
                "max_equity": 0.8,
                "min_bonds": 0.15,
                "max_alternatives": 0.2,
                "target_volatility": 0.16
            },
            RiskProfile.AGGRESSIVE: {
                "max_equity": 1.0,
                "min_bonds": 0.0,
                "max_alternatives": 0.3,
                "target_volatility": 0.22
            }
        }
        
        risk_params = risk_mappings[risk_profile]
        
        # Generate allocation based on risk profile and constraints
        # This is a simplified allocation - in reality would use optimization algorithms
        base_allocations = {
            RiskProfile.CONSERVATIVE: {
                "DNB_GLOBAL_INDEKS_S": 0.25,
                "STOREBRAND_EUROPA_A_SEK": 0.15,
                "AVANZA_USA": 0.15,
                "DNB_NORDEN_INDEKS_S": 0.10,
                "XETRA_GOLD_ETC": 0.15,
                "PLUS_FASTIGHETER_SVERIGE_INDEX": 0.10,
                "HANDELSBANKEN_GLOBAL_SMAB_INDEX": 0.10
            },
            RiskProfile.BALANCED: {
                "DNB_GLOBAL_INDEKS_S": 0.30,
                "AVANZA_USA": 0.25,
                "STOREBRAND_EUROPA_A_SEK": 0.15,
                "DNB_NORDEN_INDEKS_S": 0.10,
                "AVANZA_EMERGING_MARKETS": 0.05,
                "XETRA_GOLD_ETC": 0.10,
                "PLUS_FASTIGHETER_SVERIGE_INDEX": 0.05
            },
            RiskProfile.AGGRESSIVE: {
                "AVANZA_USA": 0.35,
                "DNB_GLOBAL_INDEKS_S": 0.25,
                "AVANZA_EMERGING_MARKETS": 0.15,
                "STOREBRAND_EUROPA_A_SEK": 0.10,
                "DNB_NORDEN_INDEKS_S": 0.05,
                "HANDELSBANKEN_GLOBAL_SMAB_INDEX": 0.05,
                "VIRTUNE_BITCOIN_PRIME_ETP": 0.05
            }
        }
        
        allocation = base_allocations[risk_profile].copy()
        
        # Apply constraints if provided
        if constraints:
            allocation = self._apply_constraints(allocation, constraints)
        
        # Adjust for investment horizon
        if investment_horizon > 15:
            # Longer horizon - can take more risk
            allocation = self._adjust_for_long_horizon(allocation, risk_profile)
        elif investment_horizon < 5:
            # Shorter horizon - reduce risk
            allocation = self._adjust_for_short_horizon(allocation, risk_profile)
        
        return {
            "allocation": allocation,
            "risk_parameters": risk_params,
            "details": {
                "optimization_method": "ai_guided_strategic",
                "constraints_applied": constraints is not None,
                "horizon_adjustment": investment_horizon,
                "user_tier": user_tier
            }
        }
    
    async def _calculate_portfolio_metrics(
        self, 
        allocation: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate expected portfolio performance metrics."""
        
        self.logger.info("Calculating portfolio metrics")
        
        # Fund return and risk assumptions (simplified)
        fund_metrics = {
            "DNB_GLOBAL_INDEKS_S": {"return": 0.08, "volatility": 0.16, "sharpe": 0.50},
            "AVANZA_USA": {"return": 0.10, "volatility": 0.16, "sharpe": 0.63},
            "STOREBRAND_EUROPA_A_SEK": {"return": 0.07, "volatility": 0.18, "sharpe": 0.39},
            "DNB_NORDEN_INDEKS_S": {"return": 0.09, "volatility": 0.20, "sharpe": 0.45},
            "AVANZA_EMERGING_MARKETS": {"return": 0.06, "volatility": 0.22, "sharpe": 0.27},
            "STOREBRAND_JAPAN_A_SEK": {"return": 0.05, "volatility": 0.19, "sharpe": 0.26},
            "HANDELSBANKEN_GLOBAL_SMAB_INDEX": {"return": 0.09, "volatility": 0.24, "sharpe": 0.38},
            "XETRA_GOLD_ETC": {"return": 0.04, "volatility": 0.16, "sharpe": 0.25},
            "VIRTUNE_BITCOIN_PRIME_ETP": {"return": 0.15, "volatility": 0.80, "sharpe": 0.19},
            "XBT_ETHER_ONE": {"return": 0.20, "volatility": 0.90, "sharpe": 0.22},
            "PLUS_FASTIGHETER_SVERIGE_INDEX": {"return": 0.08, "volatility": 0.18, "sharpe": 0.44}
        }
        
        # Calculate weighted portfolio metrics
        expected_return = sum(
            allocation.get(fund, 0) * fund_metrics.get(fund, {"return": 0.06})["return"]
            for fund in allocation.keys()
        )
        
        # Simplified volatility calculation (assumes some correlation)
        weighted_variance = sum(
            (allocation.get(fund, 0) ** 2) * (fund_metrics.get(fund, {"volatility": 0.15})["volatility"] ** 2)
            for fund in allocation.keys()
        )
        
        # Add correlation adjustment (simplified)
        correlation_adjustment = 0.85  # Assumes 85% correlation benefit
        expected_volatility = (weighted_variance * correlation_adjustment) ** 0.5
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.02  # 2% risk-free rate
        expected_sharpe = (expected_return - risk_free_rate) / expected_volatility if expected_volatility > 0 else 0
        
        # Estimate max drawdown (simplified)
        expected_max_drawdown = -expected_volatility * 2.5  # Rule of thumb
        
        # Calculate diversification ratio
        avg_fund_volatility = sum(
            allocation.get(fund, 0) * fund_metrics.get(fund, {"volatility": 0.15})["volatility"]
            for fund in allocation.keys()
        )
        diversification_ratio = avg_fund_volatility / expected_volatility if expected_volatility > 0 else 1.0
        
        return {
            "expected_annual_return": round(expected_return, 4),
            "expected_volatility": round(expected_volatility, 4),
            "expected_sharpe_ratio": round(expected_sharpe, 3),
            "expected_max_drawdown": round(expected_max_drawdown, 4),
            "diversification_ratio": round(diversification_ratio, 3),
            "value_at_risk_95": round(expected_return - 1.65 * expected_volatility, 4),
            "conditional_var_95": round(expected_return - 2.33 * expected_volatility, 4)
        }
    
    async def _perform_historical_analysis(
        self, 
        allocation: Dict[str, float]
    ) -> Dict[str, Any]:
        """Perform historical performance analysis."""
        
        self.logger.info("Performing historical analysis")
        
        # Simulate historical analysis results
        # In real implementation, this would analyze actual historical data
        
        return {
            "annualized_return": 0.082,
            "volatility": 0.156,
            "sharpe_ratio": 0.42,
            "max_drawdown": -0.286,
            "best_year_return": 0.324,
            "worst_year_return": -0.198,
            "positive_years_percentage": 0.75,
            "total_return": {
                "5y": 0.52,
                "10y": 1.34,
                "20y": 3.86
            }
        }
    
    async def _perform_stress_testing(
        self, 
        allocation: Dict[str, float]
    ) -> Dict[str, Any]:
        """Perform portfolio stress testing."""
        
        self.logger.info("Performing stress testing")
        
        # Simulate stress test results
        return {
            "scenarios": {
                "2008_crisis": {
                    "scenario_name": "2008 Financial Crisis",
                    "portfolio_return": -0.34,
                    "duration_days": 180,
                    "max_drawdown": -0.42,
                    "recovery_time_days": 720,
                    "description": "Global financial crisis impact"
                },
                "covid_2020": {
                    "scenario_name": "COVID-19 Pandemic",
                    "portfolio_return": -0.18,
                    "duration_days": 60,
                    "max_drawdown": -0.28,
                    "recovery_time_days": 180,
                    "description": "Pandemic market shock"
                }
            },
            "worst_case_loss": -0.42,
            "average_crisis_performance": -0.26,
            "recovery_time_estimate": "12-24 months",
            "risk_score": 65.5
        }
    
    async def _get_swedish_economic_context(self) -> Dict[str, Any]:
        """Get current Swedish economic context."""
        
        self.logger.info("Gathering Swedish economic context")
        
        return {
            "current_phase": "late_cycle_slowdown",
            "key_indicators": {
                "repo_rate": 4.0,
                "inflation_cpi": 2.1,
                "gdp_growth": 1.1,
                "unemployment_rate": 7.8,
                "house_price_change": -12.5
            },
            "trends": {
                "interest_rate_direction": "stable_to_higher",
                "inflation_trend": "declining",
                "growth_momentum": "weakening",
                "housing_market_trend": "correction",
                "labor_market_trend": "cooling"
            },
            "investment_implications": [
                "Higher interest rates favor defensive positioning",
                "Housing market correction may impact Swedish equities",
                "Disinflation trend supports bond allocations",
                "Economic slowdown suggests defensive fund selection"
            ],
            "confidence_score": 0.82
        }
    
    async def _analyze_market_regimes(
        self, 
        allocation: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze portfolio performance across market regimes."""
        
        self.logger.info("Analyzing market regimes")
        
        return {
            "bull_market": {
                "average_return": 0.12,
                "volatility": 0.14,
                "frequency": 0.65,
                "max_drawdown": -0.08
            },
            "bear_market": {
                "average_return": -0.15,
                "volatility": 0.22,
                "frequency": 0.20,
                "max_drawdown": -0.35
            },
            "high_volatility": {
                "average_return": 0.02,
                "volatility": 0.28,
                "frequency": 0.25,
                "max_drawdown": -0.25
            },
            "low_volatility": {
                "average_return": 0.08,
                "volatility": 0.09,
                "frequency": 0.40,
                "max_drawdown": -0.05
            },
            "crisis": {
                "average_return": -0.22,
                "volatility": 0.35,
                "frequency": 0.05,
                "max_drawdown": -0.45
            }
        }
    
    async def _generate_ai_reasoning(
        self,
        allocation_result: Dict[str, Any],
        expected_metrics: Dict[str, Any],
        swedish_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate AI-powered reasoning for the allocation."""
        
        self.logger.info("Generating AI reasoning")
        
        # Simulate AI-generated reasoning
        # In real implementation, this would call the AI service
        
        return {
            "allocation_rationale": (
                "The recommended allocation balances growth potential with risk management, "
                "emphasizing global diversification while maintaining Swedish market exposure. "
                "The allocation to US and global equity funds provides growth potential, while "
                "defensive assets like gold and real estate offer downside protection."
            ),
            "swedish_economic_rationale": (
                f"Current Swedish economic conditions (phase: {swedish_context['current_phase']}) "
                "suggest a defensive tilt is appropriate. With interest rates at 4.0% and housing "
                "market correction ongoing, the allocation favors internationally diversified funds "
                "over pure Swedish equity exposure."
            ),
            "risk_assessment": (
                f"Expected portfolio volatility of {expected_metrics['expected_volatility']:.1%} "
                f"aligns with the specified risk profile. The {expected_metrics['diversification_ratio']:.2f} "
                "diversification ratio indicates good risk spreading across asset classes and regions."
            ),
            "historical_context": (
                "Historical analysis shows this allocation would have delivered positive returns "
                "in 75% of years, with manageable drawdowns during major market crises. "
                "The allocation's defensive components helped limit losses during 2008 and 2020 crises."
            ),
            "regime_considerations": (
                "The allocation is designed to perform reasonably across different market regimes, "
                "with particular strength in low-volatility periods and defensive characteristics "
                "during crisis periods. Bull market participation is maintained through equity exposure."
            )
        }
    
    def _calculate_confidence_score(
        self,
        data_quality: Dict[str, Any],
        allocation_result: Dict[str, Any],
        expected_metrics: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for the analysis."""
        
        confidence_factors = []
        
        # Data quality factor (40% weight)
        confidence_factors.append(data_quality["overall_score"] * 0.4)
        
        # Allocation diversification factor (30% weight)
        num_funds = len(allocation_result["allocation"])
        diversification_score = min(num_funds / 6, 1.0)  # Optimal around 6 funds
        confidence_factors.append(diversification_score * 0.3)
        
        # Metrics reasonableness factor (20% weight)
        sharpe_ratio = expected_metrics["expected_sharpe_ratio"]
        sharpe_score = min(abs(sharpe_ratio) / 0.5, 1.0)  # Good Sharpe around 0.5
        confidence_factors.append(sharpe_score * 0.2)
        
        # Economic context factor (10% weight)
        economic_confidence = 0.8  # Simulated
        confidence_factors.append(economic_confidence * 0.1)
        
        total_confidence = sum(confidence_factors)
        return round(max(0.1, min(1.0, total_confidence)), 3)
    
    def _apply_constraints(
        self, 
        allocation: Dict[str, float],
        constraints: PortfolioConstraints
    ) -> Dict[str, float]:
        """Apply portfolio constraints to allocation."""
        
        # Apply max funds constraint
        if len(allocation) > constraints.max_funds:
            # Keep top funds by allocation
            sorted_funds = sorted(
                allocation.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:constraints.max_funds]
            allocation = dict(sorted_funds)
            
            # Renormalize
            total = sum(allocation.values())
            allocation = {k: v/total for k, v in allocation.items()}
        
        # Apply minimum allocation constraint
        for fund, weight in list(allocation.items()):
            if weight < constraints.min_allocation_per_fund:
                del allocation[fund]
        
        # Renormalize after removing small allocations
        total = sum(allocation.values())
        if total > 0:
            allocation = {k: v/total for k, v in allocation.items()}
        
        # Remove excluded funds
        if constraints.exclude_funds:
            for fund in constraints.exclude_funds:
                allocation.pop(fund, None)
            
            # Renormalize again
            total = sum(allocation.values())
            if total > 0:
                allocation = {k: v/total for k, v in allocation.items()}
        
        return allocation
    
    def _adjust_for_long_horizon(
        self, 
        allocation: Dict[str, float],
        risk_profile: RiskProfile
    ) -> Dict[str, float]:
        """Adjust allocation for long investment horizon."""
        
        # Increase equity allocation for long horizons
        equity_boost = 0.05 if risk_profile != RiskProfile.AGGRESSIVE else 0.02
        
        # Boost growth funds
        growth_funds = ["AVANZA_USA", "DNB_GLOBAL_INDEKS_S", "AVANZA_EMERGING_MARKETS"]
        
        for fund in growth_funds:
            if fund in allocation:
                allocation[fund] += equity_boost / len(growth_funds)
        
        # Reduce defensive allocations
        defensive_funds = ["XETRA_GOLD_ETC"]
        for fund in defensive_funds:
            if fund in allocation:
                allocation[fund] = max(0.05, allocation[fund] - equity_boost)
        
        # Renormalize
        total = sum(allocation.values())
        allocation = {k: v/total for k, v in allocation.items()}
        
        return allocation
    
    def _adjust_for_short_horizon(
        self, 
        allocation: Dict[str, float],
        risk_profile: RiskProfile
    ) -> Dict[str, float]:
        """Adjust allocation for short investment horizon."""
        
        # Increase defensive allocation for short horizons
        defensive_boost = 0.1 if risk_profile == RiskProfile.AGGRESSIVE else 0.05
        
        # Boost defensive funds
        defensive_funds = ["XETRA_GOLD_ETC", "PLUS_FASTIGHETER_SVERIGE_INDEX"]
        
        for fund in defensive_funds:
            if fund in allocation:
                allocation[fund] += defensive_boost / len(defensive_funds)
            else:
                allocation[fund] = defensive_boost / len(defensive_funds)
        
        # Reduce high-risk allocations
        high_risk_funds = ["VIRTUNE_BITCOIN_PRIME_ETP", "XBT_ETHER_ONE", "AVANZA_EMERGING_MARKETS"]
        for fund in high_risk_funds:
            if fund in allocation:
                allocation[fund] = max(0, allocation[fund] - defensive_boost / 3)
                if allocation[fund] < 0.02:  # Remove very small allocations
                    del allocation[fund]
        
        # Renormalize
        total = sum(allocation.values())
        if total > 0:
            allocation = {k: v/total for k, v in allocation.items()}
        
        return allocation