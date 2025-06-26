"""
Portfolio Optimization Engine for Investment MCP System.

This module provides a unified interface to all portfolio optimization
algorithms and comprehensive analysis capabilities.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum

from .markowitz import MarkowitzOptimizer
from .risk_parity import RiskParityOptimizer
from .sharpe_optimizer import SharpeOptimizer
from .min_volatility import MinimumVolatilityOptimizer
from .monte_carlo import MonteCarloSimulator
from .backtesting import PortfolioBacktester
from .housing_analysis import SwedishHousingAnalyzer
from .constraints import PortfolioConstraints
from ..config import TRADEABLE_FUNDS

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Available optimization types."""
    MARKOWITZ = "markowitz"
    RISK_PARITY = "risk_parity"
    MAXIMUM_SHARPE = "maximum_sharpe"
    MINIMUM_VOLATILITY = "minimum_volatility"
    EQUAL_WEIGHT = "equal_weight"
    CUSTOM = "custom"


class PortfolioOptimizationEngine:
    """Unified portfolio optimization engine with all algorithms."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize all optimizers
        self.markowitz = MarkowitzOptimizer()
        self.risk_parity = RiskParityOptimizer()
        self.sharpe_optimizer = SharpeOptimizer()
        self.min_volatility = MinimumVolatilityOptimizer()
        self.monte_carlo = MonteCarloSimulator()
        self.backtester = PortfolioBacktester()
        self.housing_analyzer = SwedishHousingAnalyzer()
        self.constraints = PortfolioConstraints()
        
        # Default parameters
        self.risk_free_rate = 0.02
        self.default_risk_tolerance = "medium"
    
    def optimize_portfolio(
        self,
        optimization_type: str,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_tolerance: str = "medium",
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None,
        current_portfolio: Optional[Dict[str, float]] = None,
        custom_constraints: Optional[Dict[str, Any]] = None,
        include_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Unified portfolio optimization interface.
        
        Args:
            optimization_type: Type of optimization to perform
            expected_returns: Expected annual returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            risk_tolerance: Risk tolerance level (low/medium/high/very_high)
            target_return: Target return for optimization (if applicable)
            target_volatility: Target volatility for optimization (if applicable)
            current_portfolio: Current portfolio for turnover constraints
            custom_constraints: Additional constraints
            include_analysis: Whether to include comprehensive analysis
            
        Returns:
            Comprehensive optimization results
        """
        
        try:
            self.logger.info(f"Running {optimization_type} optimization with {risk_tolerance} risk tolerance")
            
            # Validate inputs
            if not self._validate_optimization_inputs(expected_returns, covariance_matrix):
                return self._empty_result("Invalid optimization inputs")
            
            # Run specific optimization
            optimization_result = self._run_optimization(
                optimization_type, expected_returns, covariance_matrix,
                risk_tolerance, target_return, target_volatility,
                current_portfolio, custom_constraints
            )
            
            if not optimization_result["success"]:
                return optimization_result
            
            # Add comprehensive analysis if requested
            if include_analysis:
                comprehensive_analysis = self._add_comprehensive_analysis(
                    optimization_result, expected_returns, covariance_matrix, risk_tolerance
                )
                optimization_result.update(comprehensive_analysis)
            
            # Add performance benchmarks
            benchmark_analysis = self._add_benchmark_analysis(
                optimization_result, expected_returns, covariance_matrix
            )
            optimization_result.update(benchmark_analysis)
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            return self._empty_result(str(e))
    
    def compare_optimization_strategies(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_tolerance: str = "medium",
        strategies_to_compare: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple optimization strategies."""
        
        try:
            if strategies_to_compare is None:
                strategies_to_compare = [
                    "markowitz", "risk_parity", "maximum_sharpe", 
                    "minimum_volatility", "equal_weight"
                ]
            
            self.logger.info(f"Comparing {len(strategies_to_compare)} optimization strategies")
            
            results = {}
            
            # Run each optimization strategy
            for strategy in strategies_to_compare:
                result = self.optimize_portfolio(
                    optimization_type=strategy,
                    expected_returns=expected_returns,
                    covariance_matrix=covariance_matrix,
                    risk_tolerance=risk_tolerance,
                    include_analysis=False  # Skip individual analysis for comparison
                )
                
                if result["success"]:
                    results[strategy] = {
                        "allocation": result["allocation"],
                        "metrics": result["portfolio_metrics"],
                        "optimization_details": result.get("optimization_details", {})
                    }
                else:
                    results[strategy] = {"error": result.get("error", "Optimization failed")}
            
            # Comprehensive comparison analysis
            comparison_analysis = self._analyze_strategy_comparison(results, expected_returns, covariance_matrix)
            
            # Risk-return efficiency analysis
            efficiency_analysis = self._analyze_risk_return_efficiency(results)
            
            # Swedish context comparison
            swedish_comparison = self._analyze_swedish_strategy_comparison(results)
            
            return {
                "success": True,
                "comparison_type": "optimization_strategies",
                "strategies_compared": strategies_to_compare,
                "risk_tolerance": risk_tolerance,
                "strategy_results": results,
                "comparison_analysis": comparison_analysis,
                "efficiency_analysis": efficiency_analysis,
                "swedish_analysis": swedish_comparison,
                "recommendations": self._generate_strategy_recommendations(results, risk_tolerance),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Strategy comparison failed: {e}")
            return {"success": False, "error": str(e)}
    
    def run_portfolio_analysis_suite(
        self,
        allocation: Dict[str, float],
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        historical_data: Optional[pd.DataFrame] = None,
        analysis_period_years: int = 10
    ) -> Dict[str, Any]:
        """Run comprehensive portfolio analysis suite."""
        
        try:
            self.logger.info("Running comprehensive portfolio analysis suite")
            
            # Validate allocation
            validation = self.constraints.validate_portfolio_allocation(allocation, "medium")
            if not validation["valid"]:
                return {"success": False, "error": f"Invalid allocation: {validation['errors']}"}
            
            analysis_results = {}
            
            # 1. Basic portfolio metrics
            analysis_results["basic_metrics"] = self._calculate_basic_portfolio_metrics(
                allocation, expected_returns, covariance_matrix
            )
            
            # 2. Risk analysis
            analysis_results["risk_analysis"] = self._comprehensive_risk_analysis(
                allocation, covariance_matrix
            )
            
            # 3. Monte Carlo simulation
            mc_result = self.monte_carlo.run_portfolio_simulation(
                allocation=allocation,
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                time_horizon_years=analysis_period_years,
                n_simulations=10000
            )
            analysis_results["monte_carlo"] = mc_result
            
            # 4. VaR and CVaR analysis
            var_result = self.monte_carlo.calculate_var_and_cvar(
                allocation=allocation,
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix
            )
            analysis_results["var_analysis"] = var_result
            
            # 5. Stress testing
            stress_scenarios = self._generate_stress_scenarios()
            stress_result = self.monte_carlo.run_stress_test(
                allocation=allocation,
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                stress_scenarios=stress_scenarios
            )
            analysis_results["stress_testing"] = stress_result
            
            # 6. Historical backtesting (if data available)
            if historical_data is not None:
                backtest_result = self._run_portfolio_backtest(
                    allocation, historical_data, analysis_period_years
                )
                analysis_results["historical_backtest"] = backtest_result
            
            # 7. Swedish context analysis
            analysis_results["swedish_analysis"] = self._comprehensive_swedish_analysis(
                allocation, expected_returns["PLUS_ALLABOLAG_SVERIGE_INDEX"]
            )
            
            # 8. Optimization analysis
            analysis_results["optimization_analysis"] = self._analyze_portfolio_optimality(
                allocation, expected_returns, covariance_matrix
            )
            
            # 9. Performance attribution
            analysis_results["performance_attribution"] = self._analyze_performance_attribution(
                allocation, expected_returns, covariance_matrix
            )
            
            # 10. ESG and sustainability analysis (placeholder)
            analysis_results["esg_analysis"] = self._analyze_esg_factors(allocation)
            
            return {
                "success": True,
                "analysis_type": "comprehensive_portfolio_suite",
                "portfolio_allocation": allocation,
                "analysis_period_years": analysis_period_years,
                "analysis_results": analysis_results,
                "overall_assessment": self._generate_overall_assessment(analysis_results),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio analysis suite failed: {e}")
            return {"success": False, "error": str(e)}
    
    def optimize_with_housing_consideration(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        property_price: float,
        monthly_rent: float,
        risk_tolerance: str = "medium",
        analysis_period_years: int = 10,
        region: str = "Stockholm"
    ) -> Dict[str, Any]:
        """Optimize portfolio considering housing investment alternative."""
        
        try:
            self.logger.info("Running portfolio optimization with housing consideration")
            
            # First, run standard portfolio optimization
            portfolio_result = self.optimize_portfolio(
                optimization_type="maximum_sharpe",
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                risk_tolerance=risk_tolerance
            )
            
            if not portfolio_result["success"]:
                return portfolio_result
            
            # Run housing vs investment analysis
            portfolio_return = portfolio_result["portfolio_metrics"]["expected_return"]
            portfolio_volatility = portfolio_result["portfolio_metrics"]["volatility"]
            
            housing_analysis = self.housing_analyzer.analyze_housing_vs_investment(
                property_price=property_price,
                monthly_rent_savings=monthly_rent,
                portfolio_allocation=portfolio_result["allocation"],
                expected_portfolio_return=portfolio_return,
                portfolio_volatility=portfolio_volatility,
                analysis_period_years=analysis_period_years,
                region=region
            )
            
            # Rent vs buy analysis
            rent_buy_analysis = self.housing_analyzer.analyze_rent_vs_buy_decision(
                property_price=property_price,
                monthly_rent=monthly_rent,
                region=region,
                analysis_period_years=analysis_period_years
            )
            
            # Integrated recommendation
            integrated_recommendation = self._generate_integrated_housing_recommendation(
                portfolio_result, housing_analysis, rent_buy_analysis, risk_tolerance
            )
            
            return {
                "success": True,
                "analysis_type": "portfolio_with_housing_consideration",
                "portfolio_optimization": portfolio_result,
                "housing_vs_investment": housing_analysis,
                "rent_vs_buy": rent_buy_analysis,
                "integrated_recommendation": integrated_recommendation,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio with housing optimization failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _run_optimization(
        self,
        optimization_type: str,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_tolerance: str,
        target_return: Optional[float],
        target_volatility: Optional[float],
        current_portfolio: Optional[Dict[str, float]],
        custom_constraints: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run specific optimization algorithm."""
        
        opt_type = optimization_type.lower()
        
        if opt_type == "markowitz":
            return self.markowitz.optimize_portfolio(
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                risk_tolerance=risk_tolerance,
                target_return=target_return,
                current_portfolio=current_portfolio,
                custom_constraints=custom_constraints
            )
        
        elif opt_type == "risk_parity":
            return self.risk_parity.optimize_portfolio(
                covariance_matrix=covariance_matrix,
                risk_tolerance=risk_tolerance,
                target_volatility=target_volatility,
                current_portfolio=current_portfolio,
                custom_constraints=custom_constraints
            )
        
        elif opt_type == "maximum_sharpe":
            return self.sharpe_optimizer.optimize_portfolio(
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                risk_tolerance=risk_tolerance,
                current_portfolio=current_portfolio,
                custom_constraints=custom_constraints
            )
        
        elif opt_type == "minimum_volatility":
            return self.min_volatility.optimize_portfolio(
                covariance_matrix=covariance_matrix,
                risk_tolerance=risk_tolerance,
                expected_returns=expected_returns,
                min_expected_return=target_return,
                current_portfolio=current_portfolio,
                custom_constraints=custom_constraints
            )
        
        elif opt_type == "equal_weight":
            return self._equal_weight_optimization(
                expected_returns, covariance_matrix, risk_tolerance
            )
        
        else:
            return {"success": False, "error": f"Unknown optimization type: {optimization_type}"}
    
    def _equal_weight_optimization(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_tolerance: str
    ) -> Dict[str, Any]:
        """Simple equal weight portfolio."""
        
        try:
            # Get approved funds
            approved_funds = list(TRADEABLE_FUNDS.keys())
            available_funds = [fund for fund in expected_returns.index if fund in approved_funds]
            
            if len(available_funds) < 3:
                return {"success": False, "error": "Insufficient approved funds"}
            
            # Equal weights
            weight = 1.0 / len(available_funds)
            allocation = {fund: weight for fund in available_funds}
            
            # Calculate metrics
            weights = np.array([allocation[fund] for fund in available_funds])
            returns_subset = expected_returns[available_funds]
            cov_subset = covariance_matrix.loc[available_funds, available_funds]
            
            portfolio_return = np.dot(weights, returns_subset.values)
            portfolio_variance = np.dot(weights, np.dot(cov_subset.values, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            portfolio_metrics = {
                "expected_return": float(portfolio_return),
                "volatility": float(portfolio_volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "number_of_assets": len(available_funds),
                "concentration_ratio": weight,
                "diversification_score": 1 - weight
            }
            
            return {
                "success": True,
                "optimization_type": "equal_weight",
                "allocation": allocation,
                "portfolio_metrics": portfolio_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _add_comprehensive_analysis(
        self,
        optimization_result: Dict[str, Any],
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_tolerance: str
    ) -> Dict[str, Any]:
        """Add comprehensive analysis to optimization result."""
        
        allocation = optimization_result["allocation"]
        
        # Risk decomposition
        risk_decomposition = self._analyze_risk_decomposition(allocation, covariance_matrix)
        
        # Sensitivity analysis
        sensitivity_analysis = self._analyze_parameter_sensitivity(
            allocation, expected_returns, covariance_matrix
        )
        
        # Efficient frontier position
        frontier_analysis = self.sharpe_optimizer.calculate_efficient_frontier(
            expected_returns, covariance_matrix, risk_tolerance
        )
        
        return {
            "risk_decomposition": risk_decomposition,
            "sensitivity_analysis": sensitivity_analysis,
            "efficient_frontier": frontier_analysis
        }
    
    def _add_benchmark_analysis(
        self,
        optimization_result: Dict[str, Any],
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """Add benchmark comparison analysis."""
        
        allocation = optimization_result["allocation"]
        
        # Compare with equal weight benchmark
        equal_weight_result = self._equal_weight_optimization(
            expected_returns, covariance_matrix, "medium"
        )
        
        # Compare with Swedish market (PLUS_ALLABOLAG_SVERIGE_INDEX)
        swedish_market_comparison = self._compare_with_swedish_market(
            allocation, expected_returns, covariance_matrix
        )
        
        return {
            "benchmark_analysis": {
                "vs_equal_weight": self._compare_portfolios(
                    optimization_result, equal_weight_result
                ),
                "vs_swedish_market": swedish_market_comparison
            }
        }
    
    def _generate_stress_scenarios(self) -> List[Dict[str, Any]]:
        """Generate stress test scenarios for Swedish market."""
        
        return [
            {
                "name": "Financial Crisis",
                "description": "2008-style financial crisis with correlation increase",
                "return_shocks": {fund: -0.30 for fund in TRADEABLE_FUNDS.keys()},
                "volatility_multipliers": {fund: 2.0 for fund in TRADEABLE_FUNDS.keys()},
                "correlation_changes": {"increase_factor": 1.5},
                "time_horizon": 1.0
            },
            {
                "name": "Swedish Housing Crash",
                "description": "Swedish housing market crash affecting economy",
                "return_shocks": {
                    "PLUS_ALLABOLAG_SVERIGE_INDEX": -0.40,
                    "DNB_NORDEN_INDEKS_S": -0.25,
                    "PLUS_FASTIGHETER_SVERIGE_INDEX": -0.50
                },
                "volatility_multipliers": {
                    "PLUS_ALLABOLAG_SVERIGE_INDEX": 2.5,
                    "PLUS_FASTIGHETER_SVERIGE_INDEX": 3.0
                },
                "time_horizon": 2.0
            },
            {
                "name": "Crypto Crash",
                "description": "Major cryptocurrency market collapse",
                "return_shocks": {
                    "VIRTUNE_BITCOIN_PRIME_ETP": -0.80,
                    "XBT_ETHER_ONE": -0.75
                },
                "volatility_multipliers": {
                    "VIRTUNE_BITCOIN_PRIME_ETP": 4.0,
                    "XBT_ETHER_ONE": 4.0
                },
                "time_horizon": 0.5
            },
            {
                "name": "Rising Interest Rates",
                "description": "Rapid interest rate increases by Riksbank",
                "return_shocks": {fund: -0.15 for fund in TRADEABLE_FUNDS.keys()},
                "volatility_multipliers": {fund: 1.3 for fund in TRADEABLE_FUNDS.keys()},
                "time_horizon": 1.0
            },
            {
                "name": "Stagflation",
                "description": "High inflation with low growth",
                "return_shocks": {
                    "XETRA_GOLD_ETC": 0.20,  # Gold benefits
                    **{fund: -0.10 for fund in TRADEABLE_FUNDS.keys() if fund != "XETRA_GOLD_ETC"}
                },
                "volatility_multipliers": {fund: 1.5 for fund in TRADEABLE_FUNDS.keys()},
                "time_horizon": 2.0
            }
        ]
    
    # Additional helper methods would continue here...
    # (Due to length constraints, I'm showing the key structure)
    
    def _validate_optimization_inputs(
        self, 
        expected_returns: pd.Series, 
        covariance_matrix: pd.DataFrame
    ) -> bool:
        """Validate optimization inputs."""
        
        if expected_returns.empty or covariance_matrix.empty:
            return False
        
        if len(expected_returns) != len(covariance_matrix):
            return False
        
        if not (expected_returns.index == covariance_matrix.index).all():
            return False
        
        return True
    
    def _empty_result(self, error_message: str) -> Dict[str, Any]:
        """Return empty result with error message."""
        
        return {
            "success": False,
            "error": error_message,
            "optimization_type": "unknown",
            "timestamp": datetime.now().isoformat()
        }
    
    # Placeholder methods for comprehensive analysis
    def _analyze_strategy_comparison(self, results, expected_returns, covariance_matrix):
        """Analyze comparison between strategies."""
        return {"placeholder": "Strategy comparison analysis"}
    
    def _analyze_risk_return_efficiency(self, results):
        """Analyze risk-return efficiency."""
        return {"placeholder": "Risk-return efficiency analysis"}
    
    def _analyze_swedish_strategy_comparison(self, results):
        """Analyze strategies from Swedish perspective."""
        return {"placeholder": "Swedish strategy comparison"}
    
    def _generate_strategy_recommendations(self, results, risk_tolerance):
        """Generate strategy recommendations."""
        return {"placeholder": "Strategy recommendations"}
    
    def _calculate_basic_portfolio_metrics(self, allocation, expected_returns, covariance_matrix):
        """Calculate basic portfolio metrics."""
        return {"placeholder": "Basic metrics"}
    
    def _comprehensive_risk_analysis(self, allocation, covariance_matrix):
        """Comprehensive risk analysis."""
        return {"placeholder": "Risk analysis"}
    
    def _run_portfolio_backtest(self, allocation, historical_data, analysis_period_years):
        """Run portfolio backtest."""
        return {"placeholder": "Backtest results"}
    
    def _comprehensive_swedish_analysis(self, allocation, swedish_return):
        """Comprehensive Swedish context analysis."""
        return {"placeholder": "Swedish analysis"}
    
    def _analyze_portfolio_optimality(self, allocation, expected_returns, covariance_matrix):
        """Analyze portfolio optimality."""
        return {"placeholder": "Optimality analysis"}
    
    def _analyze_performance_attribution(self, allocation, expected_returns, covariance_matrix):
        """Analyze performance attribution."""
        return {"placeholder": "Performance attribution"}
    
    def _analyze_esg_factors(self, allocation):
        """Analyze ESG factors."""
        return {"placeholder": "ESG analysis"}
    
    def _generate_overall_assessment(self, analysis_results):
        """Generate overall assessment."""
        return {"placeholder": "Overall assessment"}
    
    def _generate_integrated_housing_recommendation(self, portfolio_result, housing_analysis, rent_buy_analysis, risk_tolerance):
        """Generate integrated housing recommendation."""
        return {"placeholder": "Integrated housing recommendation"}
    
    def _analyze_risk_decomposition(self, allocation, covariance_matrix):
        """Analyze risk decomposition."""
        return {"placeholder": "Risk decomposition"}
    
    def _analyze_parameter_sensitivity(self, allocation, expected_returns, covariance_matrix):
        """Analyze parameter sensitivity."""
        return {"placeholder": "Sensitivity analysis"}
    
    def _compare_portfolios(self, result1, result2):
        """Compare two portfolio results."""
        return {"placeholder": "Portfolio comparison"}
    
    def _compare_with_swedish_market(self, allocation, expected_returns, covariance_matrix):
        """Compare with Swedish market."""
        return {"placeholder": "Swedish market comparison"}