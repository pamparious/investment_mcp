"""
Unified portfolio optimization module for Investment MCP System.

This module consolidates all portfolio optimization functionality into a single,
comprehensive system using Modern Portfolio Theory and AI-enhanced methods.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy.optimize import minimize
import warnings
from ..core.config import TRADEABLE_FUNDS, validate_fund_allocation

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Comprehensive portfolio optimization using multiple approaches."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        self.approved_funds = list(TRADEABLE_FUNDS.keys())
    
    def mean_variance_optimization(
        self, 
        returns_matrix: pd.DataFrame,
        target_return: Optional[float] = None,
        risk_tolerance: str = "medium",
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform mean-variance optimization using Modern Portfolio Theory.
        
        Args:
            returns_matrix: DataFrame with asset returns
            target_return: Target annual return (if None, optimizes for max Sharpe)
            risk_tolerance: Risk tolerance level ('low', 'medium', 'high', 'very_high')
            constraints: Additional constraints for optimization
            
        Returns:
            Dictionary containing optimal weights and portfolio metrics
        """
        
        if returns_matrix.empty:
            return self._empty_optimization_result()
        
        try:
            # Filter to approved funds only
            available_funds = [col for col in returns_matrix.columns if col in self.approved_funds]
            if not available_funds:
                self.logger.error("No approved funds found in returns matrix")
                return self._empty_optimization_result()
            
            returns_data = returns_matrix[available_funds].dropna()
            if len(returns_data) < 30:  # Need sufficient data
                self.logger.warning("Insufficient data for optimization")
                return self._empty_optimization_result()
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns_data.mean() * 252  # Annualized
            cov_matrix = returns_data.cov() * 252  # Annualized
            
            n_assets = len(available_funds)
            
            # Set up optimization constraints
            opt_constraints = self._get_optimization_constraints(n_assets, risk_tolerance, constraints)
            
            # Define objective function
            if target_return is not None:
                # Minimize risk for target return
                def objective(weights):
                    return np.dot(weights, np.dot(cov_matrix.values, weights))
                
                # Add return constraint
                return_constraint = {
                    'type': 'eq',
                    'fun': lambda weights: np.dot(weights, expected_returns.values) - target_return
                }
                opt_constraints.append(return_constraint)
            else:
                # Maximize Sharpe ratio (risk-free rate = 0)
                def objective(weights):
                    portfolio_return = np.dot(weights, expected_returns.values)
                    portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))
                    return -portfolio_return / portfolio_risk if portfolio_risk > 0 else -1000
            
            # Initial guess (equal weights)
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Bounds for weights (0 to max allocation based on risk tolerance)
            max_allocation = self._get_max_allocation(risk_tolerance)
            bounds = tuple((0, max_allocation) for _ in range(n_assets))
            
            # Perform optimization
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=opt_constraints,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                self.logger.warning(f"Optimization failed: {result.message}")
                return self._get_fallback_allocation(available_funds, risk_tolerance)
            
            # Create results
            optimal_weights = result.x
            fund_allocation = dict(zip(available_funds, optimal_weights))
            
            # Validate allocation
            validation = validate_fund_allocation(fund_allocation)
            if not validation["valid"]:
                self.logger.warning(f"Invalid allocation: {validation['errors']}")
                return self._get_fallback_allocation(available_funds, risk_tolerance)
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(optimal_weights, expected_returns.values)
            portfolio_risk = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix.values, optimal_weights)))
            sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
            
            # Calculate additional metrics
            portfolio_returns = np.dot(returns_data.values, optimal_weights)
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            
            return {
                "optimization_type": "mean_variance",
                "success": True,
                "fund_allocation": fund_allocation,
                "portfolio_metrics": {
                    "expected_return": float(portfolio_return),
                    "expected_volatility": float(portfolio_risk),
                    "sharpe_ratio": float(sharpe_ratio),
                    "max_drawdown": float(max_drawdown)
                },
                "risk_tolerance": risk_tolerance,
                "number_of_funds": len([w for w in optimal_weights if w > 0.01]),
                "concentration": float(max(optimal_weights)),
                "optimization_details": {
                    "iterations": result.nit,
                    "status": result.message
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in mean-variance optimization: {e}")
            return self._get_fallback_allocation(self.approved_funds[:5], risk_tolerance)
    
    def risk_parity_optimization(self, returns_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform risk parity optimization where each asset contributes equally to portfolio risk.
        
        Args:
            returns_matrix: DataFrame with asset returns
            
        Returns:
            Dictionary containing risk parity weights and portfolio metrics
        """
        
        if returns_matrix.empty:
            return self._empty_optimization_result()
        
        try:
            # Filter to approved funds
            available_funds = [col for col in returns_matrix.columns if col in self.approved_funds]
            if not available_funds:
                return self._empty_optimization_result()
            
            returns_data = returns_matrix[available_funds].dropna()
            if len(returns_data) < 30:
                return self._empty_optimization_result()
            
            # Calculate covariance matrix
            cov_matrix = returns_data.cov() * 252  # Annualized
            n_assets = len(available_funds)
            
            # Objective function: minimize sum of squared differences in risk contribution
            def objective(weights):
                portfolio_variance = np.dot(weights, np.dot(cov_matrix.values, weights))
                marginal_contrib = np.dot(cov_matrix.values, weights)
                risk_contrib = weights * marginal_contrib / portfolio_variance
                target_contrib = 1.0 / n_assets
                return np.sum((risk_contrib - target_contrib) ** 2)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0}  # Weights sum to 1
            ]
            
            # Bounds
            bounds = tuple((0.01, 0.5) for _ in range(n_assets))  # Min 1%, max 50%
            
            # Initial guess
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                self.logger.warning("Risk parity optimization failed, using equal weights")
                optimal_weights = np.array([1.0 / n_assets] * n_assets)
            else:
                optimal_weights = result.x
            
            fund_allocation = dict(zip(available_funds, optimal_weights))
            
            # Calculate portfolio metrics
            expected_returns = returns_data.mean() * 252
            portfolio_return = np.dot(optimal_weights, expected_returns.values)
            portfolio_risk = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix.values, optimal_weights)))
            sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
            
            # Calculate risk contributions
            marginal_contrib = np.dot(cov_matrix.values, optimal_weights)
            risk_contrib = optimal_weights * marginal_contrib / np.dot(optimal_weights, marginal_contrib)
            
            return {
                "optimization_type": "risk_parity",
                "success": True,
                "fund_allocation": fund_allocation,
                "portfolio_metrics": {
                    "expected_return": float(portfolio_return),
                    "expected_volatility": float(portfolio_risk),
                    "sharpe_ratio": float(sharpe_ratio)
                },
                "risk_contributions": dict(zip(available_funds, risk_contrib)),
                "number_of_funds": len(available_funds),
                "concentration": float(max(optimal_weights))
            }
            
        except Exception as e:
            self.logger.error(f"Error in risk parity optimization: {e}")
            return self._empty_optimization_result()
    
    def factor_based_optimization(
        self, 
        returns_matrix: pd.DataFrame,
        factors: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Perform factor-based optimization considering fund categories and factors.
        
        Args:
            returns_matrix: DataFrame with asset returns
            factors: Factor exposures/preferences
            
        Returns:
            Dictionary containing factor-based allocation
        """
        
        if factors is None:
            factors = {
                "global_equity": 0.3,
                "regional_equity": 0.3,
                "alternative": 0.2,
                "defensive": 0.2
            }
        
        try:
            # Categorize funds by type
            fund_categories = self._categorize_funds()
            
            # Start with strategic allocation
            strategic_allocation = {}
            
            for category, target_weight in factors.items():
                category_funds = fund_categories.get(category, [])
                available_category_funds = [f for f in category_funds if f in returns_matrix.columns]
                
                if available_category_funds:
                    # Equal weight within category
                    weight_per_fund = target_weight / len(available_category_funds)
                    for fund in available_category_funds:
                        strategic_allocation[fund] = weight_per_fund
            
            # Normalize weights to sum to 1
            total_weight = sum(strategic_allocation.values())
            if total_weight > 0:
                strategic_allocation = {k: v / total_weight for k, v in strategic_allocation.items()}
            
            # Calculate expected metrics
            if strategic_allocation and not returns_matrix.empty:
                available_funds = list(strategic_allocation.keys())
                returns_data = returns_matrix[available_funds].dropna()
                
                if not returns_data.empty:
                    weights = np.array([strategic_allocation[fund] for fund in available_funds])
                    expected_returns = returns_data.mean() * 252
                    cov_matrix = returns_data.cov() * 252
                    
                    portfolio_return = np.dot(weights, expected_returns.values)
                    portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))
                    sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
                    
                    return {
                        "optimization_type": "factor_based",
                        "success": True,
                        "fund_allocation": strategic_allocation,
                        "portfolio_metrics": {
                            "expected_return": float(portfolio_return),
                            "expected_volatility": float(portfolio_risk),
                            "sharpe_ratio": float(sharpe_ratio)
                        },
                        "factor_exposures": factors,
                        "fund_categories": fund_categories
                    }
            
            return self._empty_optimization_result()
            
        except Exception as e:
            self.logger.error(f"Error in factor-based optimization: {e}")
            return self._empty_optimization_result()
    
    def create_model_portfolios(self) -> Dict[str, Dict[str, Any]]:
        """Create standard model portfolios for different risk profiles."""
        
        try:
            model_portfolios = {
                "conservative": {
                    "description": "Low risk portfolio for capital preservation",
                    "risk_tolerance": "low",
                    "fund_allocation": {
                        "DNB_GLOBAL_INDEKS_S": 0.25,
                        "PLUS_ALLABOLAG_SVERIGE_INDEX": 0.25,
                        "XETRA_GOLD_ETC": 0.20,
                        "PLUS_FASTIGHETER_SVERIGE_INDEX": 0.15,
                        "STOREBRAND_EUROPA_A_SEK": 0.15
                    },
                    "target_return": 0.06,
                    "expected_volatility": 0.08
                },
                "balanced": {
                    "description": "Moderate risk portfolio balancing growth and stability",
                    "risk_tolerance": "medium",
                    "fund_allocation": {
                        "DNB_GLOBAL_INDEKS_S": 0.25,
                        "AVANZA_USA": 0.20,
                        "PLUS_ALLABOLAG_SVERIGE_INDEX": 0.15,
                        "STOREBRAND_EUROPA_A_SEK": 0.15,
                        "AVANZA_EMERGING_MARKETS": 0.10,
                        "DNB_NORDEN_INDEKS_S": 0.10,
                        "XETRA_GOLD_ETC": 0.05
                    },
                    "target_return": 0.08,
                    "expected_volatility": 0.12
                },
                "growth": {
                    "description": "Higher risk portfolio focused on long-term growth",
                    "risk_tolerance": "high",
                    "fund_allocation": {
                        "AVANZA_USA": 0.25,
                        "DNB_GLOBAL_INDEKS_S": 0.20,
                        "HANDELSBANKEN_GLOBAL_SMAB_INDEX": 0.15,
                        "AVANZA_EMERGING_MARKETS": 0.15,
                        "STOREBRAND_EUROPA_A_SEK": 0.10,
                        "STOREBRAND_JAPAN_A_SEK": 0.10,
                        "VIRTUNE_BITCOIN_PRIME_ETP": 0.05
                    },
                    "target_return": 0.10,
                    "expected_volatility": 0.16
                },
                "aggressive": {
                    "description": "High risk portfolio for maximum growth potential",
                    "risk_tolerance": "very_high",
                    "fund_allocation": {
                        "AVANZA_USA": 0.20,
                        "HANDELSBANKEN_GLOBAL_SMAB_INDEX": 0.20,
                        "AVANZA_EMERGING_MARKETS": 0.20,
                        "DNB_GLOBAL_INDEKS_S": 0.15,
                        "VIRTUNE_BITCOIN_PRIME_ETP": 0.10,
                        "XBT_ETHER_ONE": 0.10,
                        "STOREBRAND_JAPAN_A_SEK": 0.05
                    },
                    "target_return": 0.12,
                    "expected_volatility": 0.20
                }
            }
            
            # Validate all portfolios
            for portfolio_name, portfolio in model_portfolios.items():
                validation = validate_fund_allocation(portfolio["fund_allocation"])
                if not validation["valid"]:
                    self.logger.error(f"Invalid model portfolio {portfolio_name}: {validation['errors']}")
            
            return model_portfolios
            
        except Exception as e:
            self.logger.error(f"Error creating model portfolios: {e}")
            return {}
    
    def _get_optimization_constraints(
        self, 
        n_assets: int, 
        risk_tolerance: str,
        additional_constraints: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get optimization constraints based on risk tolerance."""
        
        constraints = [
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0}  # Weights sum to 1
        ]
        
        # Add minimum diversification constraint
        min_assets = 3 if risk_tolerance in ["low", "medium"] else 2
        if n_assets >= min_assets:
            constraints.append({
                'type': 'ineq',
                'fun': lambda weights: np.sum(weights > 0.01) - min_assets
            })
        
        if additional_constraints:
            constraints.extend(additional_constraints.get("constraints", []))
        
        return constraints
    
    def _get_max_allocation(self, risk_tolerance: str) -> float:
        """Get maximum allocation per asset based on risk tolerance."""
        
        max_allocations = {
            "low": 0.4,
            "medium": 0.5,
            "high": 0.6,
            "very_high": 0.8
        }
        
        return max_allocations.get(risk_tolerance, 0.5)
    
    def _categorize_funds(self) -> Dict[str, List[str]]:
        """Categorize approved funds by investment type."""
        
        return {
            "global_equity": ["DNB_GLOBAL_INDEKS_S", "HANDELSBANKEN_GLOBAL_SMAB_INDEX"],
            "regional_equity": [
                "PLUS_ALLABOLAG_SVERIGE_INDEX", "DNB_NORDEN_INDEKS_S", 
                "STOREBRAND_EUROPA_A_SEK", "AVANZA_USA", "STOREBRAND_JAPAN_A_SEK",
                "AVANZA_EMERGING_MARKETS"
            ],
            "alternative": [
                "XETRA_GOLD_ETC", "VIRTUNE_BITCOIN_PRIME_ETP", "XBT_ETHER_ONE"
            ],
            "defensive": ["PLUS_FASTIGHETER_SVERIGE_INDEX"]
        }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        
        if len(returns) == 0:
            return 0.0
        
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        
        return float(np.min(drawdown))
    
    def _get_fallback_allocation(self, available_funds: List[str], risk_tolerance: str) -> Dict[str, Any]:
        """Get fallback allocation when optimization fails."""
        
        if not available_funds:
            return self._empty_optimization_result()
        
        # Create simple diversified allocation based on risk tolerance
        if risk_tolerance == "low":
            # Conservative allocation
            allocation = {fund: 1.0 / len(available_funds) for fund in available_funds[:4]}
        elif risk_tolerance == "very_high":
            # Aggressive allocation
            allocation = {fund: 1.0 / len(available_funds) for fund in available_funds}
        else:
            # Moderate allocation
            allocation = {fund: 1.0 / len(available_funds) for fund in available_funds[:6]}
        
        return {
            "optimization_type": "fallback",
            "success": True,
            "fund_allocation": allocation,
            "portfolio_metrics": {
                "expected_return": 0.07,  # Placeholder
                "expected_volatility": 0.12,  # Placeholder
                "sharpe_ratio": 0.58  # Placeholder
            },
            "risk_tolerance": risk_tolerance,
            "note": "Fallback allocation used due to optimization failure"
        }
    
    def _empty_optimization_result(self) -> Dict[str, Any]:
        """Return empty optimization result."""
        
        return {
            "optimization_type": "none",
            "success": False,
            "fund_allocation": {},
            "portfolio_metrics": {
                "expected_return": 0.0,
                "expected_volatility": 0.0,
                "sharpe_ratio": 0.0
            },
            "error": "Optimization failed"
        }