"""
Markowitz Mean-Variance Optimization for Investment MCP System.

This module implements the classic Markowitz portfolio optimization
with constraints specific to Swedish investment requirements.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Tuple
from scipy.optimize import minimize
from datetime import datetime

from ..config import TRADEABLE_FUNDS
from .constraints import PortfolioConstraints

logger = logging.getLogger(__name__)


class MarkowitzOptimizer:
    """Markowitz mean-variance portfolio optimization implementation."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.constraints = PortfolioConstraints()
        
        # Swedish market specific parameters
        self.risk_free_rate = 0.02  # Approximate Swedish 10-year bond rate
        self.trading_costs = 0.0025  # 0.25% trading costs
    
    def optimize_portfolio(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_tolerance: str,
        target_return: Optional[float] = None,
        current_portfolio: Optional[Dict[str, float]] = None,
        custom_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform Markowitz mean-variance optimization.
        
        Args:
            expected_returns: Expected annual returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            risk_tolerance: Risk tolerance level (low/medium/high/very_high)
            target_return: Target annual return (if None, maximizes Sharpe ratio)
            current_portfolio: Current portfolio weights for turnover constraints
            custom_constraints: Additional custom constraints
            
        Returns:
            Dictionary containing optimization results
        """
        
        try:
            # Validate inputs
            if not self._validate_inputs(expected_returns, covariance_matrix):
                return self._empty_result("Invalid inputs")
            
            # Filter to approved funds only
            approved_funds = list(TRADEABLE_FUNDS.keys())
            available_funds = [fund for fund in expected_returns.index if fund in approved_funds]
            
            if len(available_funds) < 3:
                return self._empty_result("Insufficient approved funds available")
            
            # Subset data to available funds
            returns = expected_returns[available_funds]
            cov_matrix = covariance_matrix.loc[available_funds, available_funds]
            
            n_assets = len(available_funds)
            
            # Set up optimization problem
            if target_return is not None:
                # Minimize risk for target return
                result = self._minimize_risk_for_return(
                    returns, cov_matrix, target_return, risk_tolerance, 
                    current_portfolio, custom_constraints
                )
            else:
                # Maximize Sharpe ratio
                result = self._maximize_sharpe_ratio(
                    returns, cov_matrix, risk_tolerance,
                    current_portfolio, custom_constraints
                )
            
            if result["success"]:
                # Create portfolio allocation
                weights = result["weights"]
                allocation = dict(zip(available_funds, weights))
                
                # Calculate portfolio metrics
                portfolio_metrics = self._calculate_portfolio_metrics(
                    allocation, returns, cov_matrix
                )
                
                # Add Swedish-specific analysis
                swedish_analysis = self._analyze_swedish_context(allocation)
                
                return {
                    "success": True,
                    "optimization_type": "markowitz",
                    "allocation": allocation,
                    "portfolio_metrics": portfolio_metrics,
                    "swedish_analysis": swedish_analysis,
                    "optimization_details": {
                        "target_return": target_return,
                        "risk_tolerance": risk_tolerance,
                        "n_iterations": result.get("n_iterations", 0),
                        "optimization_status": result.get("status", "unknown"),
                        "objective_value": result.get("objective_value", 0)
                    },
                    "constraints_applied": self._get_applied_constraints(risk_tolerance),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return self._empty_result(result.get("error", "Optimization failed"))
                
        except Exception as e:
            self.logger.error(f"Markowitz optimization failed: {e}")
            return self._empty_result(str(e))
    
    def _minimize_risk_for_return(
        self,
        returns: pd.Series,
        cov_matrix: pd.DataFrame,
        target_return: float,
        risk_tolerance: str,
        current_portfolio: Optional[Dict[str, float]] = None,
        custom_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Minimize portfolio risk for a given target return."""
        
        n_assets = len(returns)
        
        # Objective function: minimize portfolio variance
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix.values, weights))
        
        # Constraints
        constraints = []
        
        # Return constraint: portfolio return = target return
        constraints.append({
            'type': 'eq',
            'fun': lambda weights: np.dot(weights, returns.values) - target_return
        })
        
        # Weights sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda weights: np.sum(weights) - 1.0
        })
        
        # Add risk-based constraints
        risk_constraints = self.constraints.get_risk_constraints(risk_tolerance, n_assets)
        constraints.extend(risk_constraints)
        
        # Add custom constraints
        if custom_constraints:
            constraints.extend(custom_constraints.get("constraints", []))
        
        # Bounds
        bounds = self.constraints.get_weight_bounds(risk_tolerance, n_assets)
        
        # Initial guess
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Add turnover constraints if current portfolio provided
        if current_portfolio:
            current_weights = np.array([
                current_portfolio.get(fund, 0) for fund in returns.index
            ])
            turnover_constraint = self._get_turnover_constraint(
                current_weights, risk_tolerance
            )
            if turnover_constraint:
                constraints.append(turnover_constraint)
        
        # Optimize
        try:
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                return {
                    "success": True,
                    "weights": result.x,
                    "objective_value": result.fun,
                    "n_iterations": result.nit,
                    "status": result.message
                }
            else:
                return {
                    "success": False,
                    "error": f"Optimization failed: {result.message}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Optimization error: {str(e)}"
            }
    
    def _maximize_sharpe_ratio(
        self,
        returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_tolerance: str,
        current_portfolio: Optional[Dict[str, float]] = None,
        custom_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Maximize portfolio Sharpe ratio."""
        
        n_assets = len(returns)
        
        # Objective function: negative Sharpe ratio (for minimization)
        def objective(weights):
            portfolio_return = np.dot(weights, returns.values)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))
            
            if portfolio_risk <= 1e-10:  # Avoid division by zero
                return -1000
            
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
            return -sharpe_ratio  # Negative for minimization
        
        # Constraints
        constraints = []
        
        # Weights sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda weights: np.sum(weights) - 1.0
        })
        
        # Add risk-based constraints
        risk_constraints = self.constraints.get_risk_constraints(risk_tolerance, n_assets)
        constraints.extend(risk_constraints)
        
        # Add custom constraints
        if custom_constraints:
            constraints.extend(custom_constraints.get("constraints", []))
        
        # Bounds
        bounds = self.constraints.get_weight_bounds(risk_tolerance, n_assets)
        
        # Initial guess - start with equal weights
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # Add turnover constraints if current portfolio provided
        if current_portfolio:
            current_weights = np.array([
                current_portfolio.get(fund, 0) for fund in returns.index
            ])
            turnover_constraint = self._get_turnover_constraint(
                current_weights, risk_tolerance
            )
            if turnover_constraint:
                constraints.append(turnover_constraint)
        
        # Try multiple starting points for robust optimization
        best_result = None
        best_sharpe = -np.inf
        
        starting_points = [
            np.array([1.0 / n_assets] * n_assets),  # Equal weights
            np.random.dirichlet(np.ones(n_assets)),  # Random weights
            self._get_risk_based_starting_point(returns, risk_tolerance)  # Risk-based
        ]
        
        for start_point in starting_points:
            try:
                result = minimize(
                    objective,
                    start_point,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-9}
                )
                
                if result.success and -result.fun > best_sharpe:
                    best_sharpe = -result.fun
                    best_result = result
                    
            except Exception as e:
                self.logger.warning(f"Optimization attempt failed: {e}")
                continue
        
        if best_result and best_result.success:
            return {
                "success": True,
                "weights": best_result.x,
                "objective_value": best_result.fun,
                "sharpe_ratio": -best_result.fun,
                "n_iterations": best_result.nit,
                "status": best_result.message
            }
        else:
            return {
                "success": False,
                "error": "All optimization attempts failed"
            }
    
    def calculate_efficient_frontier(
        self,
        returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_tolerance: str,
        n_points: int = 20
    ) -> Dict[str, Any]:
        """Calculate the efficient frontier."""
        
        try:
            # Determine return range
            min_return = returns.min()
            max_return = returns.max()
            
            # Adjust range based on risk tolerance
            risk_multipliers = {
                "low": (0.3, 0.6),
                "medium": (0.2, 0.8),
                "high": (0.1, 0.9),
                "very_high": (0.05, 0.95)
            }
            
            low_mult, high_mult = risk_multipliers.get(risk_tolerance, (0.2, 0.8))
            return_range = max_return - min_return
            
            target_returns = np.linspace(
                min_return + low_mult * return_range,
                min_return + high_mult * return_range,
                n_points
            )
            
            frontier_points = []
            
            for target_return in target_returns:
                result = self._minimize_risk_for_return(
                    returns, cov_matrix, target_return, risk_tolerance
                )
                
                if result["success"]:
                    weights = result["weights"]
                    portfolio_return = np.dot(weights, returns.values)
                    portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))
                    sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
                    
                    frontier_points.append({
                        "target_return": target_return,
                        "actual_return": portfolio_return,
                        "risk": portfolio_risk,
                        "sharpe_ratio": sharpe_ratio,
                        "weights": weights.tolist()
                    })
            
            return {
                "success": True,
                "frontier_points": frontier_points,
                "n_points": len(frontier_points),
                "risk_tolerance": risk_tolerance,
                "calculated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Efficient frontier calculation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _calculate_portfolio_metrics(
        self,
        allocation: Dict[str, float],
        returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics."""
        
        weights = np.array([allocation.get(fund, 0) for fund in returns.index])
        
        # Basic metrics
        portfolio_return = np.dot(weights, returns.values)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix.values, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Risk-adjusted metrics
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Diversification metrics
        n_assets = len([w for w in weights if w > 0.01])  # Assets with >1% allocation
        effective_n_assets = 1 / np.sum(weights ** 2)  # Inverse of Herfindahl index
        concentration_ratio = max(weights)
        
        # Swedish-specific metrics
        swedish_allocation = sum(
            allocation.get(fund, 0) 
            for fund in allocation.keys() 
            if "SVERIGE" in fund or "NORDEN" in fund
        )
        
        return {
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "number_of_assets": int(n_assets),
            "effective_number_of_assets": float(effective_n_assets),
            "concentration_ratio": float(concentration_ratio),
            "diversification_score": float(1 - concentration_ratio),
            "swedish_allocation": float(swedish_allocation),
            "expected_annual_return_pct": float(portfolio_return * 100),
            "annual_volatility_pct": float(portfolio_volatility * 100)
        }
    
    def _analyze_swedish_context(self, allocation: Dict[str, float]) -> Dict[str, Any]:
        """Analyze portfolio from Swedish investment perspective."""
        
        # Categorize allocations
        categories = {
            "swedish_equity": 0,
            "nordic_equity": 0,
            "global_equity": 0,
            "alternatives": 0,
            "fixed_income": 0
        }
        
        for fund_id, weight in allocation.items():
            fund_info = TRADEABLE_FUNDS.get(fund_id, {})
            category = fund_info.get("category", "unknown")
            
            if "swedish" in category or "sverige" in fund_id.lower():
                categories["swedish_equity"] += weight
            elif "nordic" in category or "norden" in fund_id.lower():
                categories["nordic_equity"] += weight
            elif "global" in category or "usa" in fund_id.lower() or "europa" in fund_id.lower():
                categories["global_equity"] += weight
            elif "cryptocurrency" in category or "precious_metals" in category:
                categories["alternatives"] += weight
            elif "real_estate" in category:
                categories["fixed_income"] += weight
        
        # Swedish investment considerations
        home_bias = categories["swedish_equity"] + categories["nordic_equity"]
        currency_diversification = 1 - home_bias
        
        # Tax efficiency (Swedish perspective)
        isk_suitable_funds = sum(
            weight for fund_id, weight in allocation.items()
            if not any(crypto in fund_id.lower() for crypto in ["bitcoin", "ether"])
        )
        
        return {
            "category_breakdown": categories,
            "home_bias": float(home_bias),
            "currency_diversification": float(currency_diversification),
            "isk_suitability": float(isk_suitable_funds),
            "swedish_tax_efficiency": "high" if isk_suitable_funds > 0.8 else "medium",
            "recommendations": self._get_swedish_recommendations(categories, allocation)
        }
    
    def _get_swedish_recommendations(
        self, 
        categories: Dict[str, float], 
        allocation: Dict[str, float]
    ) -> List[str]:
        """Generate Swedish-specific investment recommendations."""
        
        recommendations = []
        
        # Home bias check
        if categories["swedish_equity"] > 0.4:
            recommendations.append("Överväg att minska svenska aktier för bättre diversifiering")
        
        # Currency diversification
        if categories["swedish_equity"] + categories["nordic_equity"] > 0.6:
            recommendations.append("Öka global exponering för valutadiversifiering")
        
        # Alternative investments
        if categories["alternatives"] > 0.15:
            recommendations.append("Hög andel alternativa investeringar - överväg riskbegränsning")
        
        # ISK suitability
        crypto_allocation = sum(
            weight for fund_id, weight in allocation.items()
            if any(crypto in fund_id.lower() for crypto in ["bitcoin", "ether"])
        )
        
        if crypto_allocation > 0.1:
            recommendations.append("Kryptovalutor passar bättre i AF-konto än ISK")
        
        if not recommendations:
            recommendations.append("Portföljen är väl diversifierad för svenska förhållanden")
        
        return recommendations
    
    def _get_turnover_constraint(
        self, 
        current_weights: np.ndarray, 
        risk_tolerance: str
    ) -> Optional[Dict[str, Any]]:
        """Get turnover constraint to limit transaction costs."""
        
        max_turnover = {
            "low": 0.3,      # Conservative investors prefer lower turnover
            "medium": 0.5,   # Moderate turnover allowed
            "high": 0.7,     # Higher turnover acceptable
            "very_high": 1.0 # No turnover constraint for aggressive investors
        }
        
        max_turn = max_turnover.get(risk_tolerance, 0.5)
        
        if max_turn >= 1.0:
            return None  # No constraint
        
        return {
            'type': 'ineq',
            'fun': lambda weights: max_turn - np.sum(np.abs(weights - current_weights))
        }
    
    def _get_risk_based_starting_point(self, returns: pd.Series, risk_tolerance: str) -> np.ndarray:
        """Get risk-based starting point for optimization."""
        
        n_assets = len(returns)
        
        if risk_tolerance == "low":
            # Start with more equal weights (conservative)
            weights = np.ones(n_assets) / n_assets
        elif risk_tolerance == "high" or risk_tolerance == "very_high":
            # Start with weights proportional to expected returns (aggressive)
            positive_returns = np.maximum(returns.values, 0.001)
            weights = positive_returns / np.sum(positive_returns)
        else:
            # Balanced approach
            inv_vol = 1 / returns.std()
            weights = inv_vol / np.sum(inv_vol)
        
        return weights
    
    def _get_applied_constraints(self, risk_tolerance: str) -> List[str]:
        """Get list of applied constraints for reporting."""
        
        constraints_list = [
            "Weights sum to 1.0",
            "Only approved Swedish funds",
            f"Risk tolerance: {risk_tolerance}"
        ]
        
        bounds = self.constraints.get_weight_bounds(risk_tolerance, 1)[0]
        constraints_list.append(f"Weight bounds: {bounds[0]:.1%} - {bounds[1]:.1%}")
        
        return constraints_list
    
    def _validate_inputs(self, returns: pd.Series, cov_matrix: pd.DataFrame) -> bool:
        """Validate optimization inputs."""
        
        if returns.empty or cov_matrix.empty:
            return False
        
        if len(returns) != len(cov_matrix):
            return False
        
        if not (returns.index == cov_matrix.index).all():
            return False
        
        if not (returns.index == cov_matrix.columns).all():
            return False
        
        # Check for NaN or infinite values
        if returns.isna().any() or cov_matrix.isna().any().any():
            return False
        
        if np.isinf(returns.values).any() or np.isinf(cov_matrix.values).any():
            return False
        
        return True
    
    def _empty_result(self, error_message: str) -> Dict[str, Any]:
        """Return empty result with error message."""
        
        return {
            "success": False,
            "error": error_message,
            "optimization_type": "markowitz",
            "allocation": {},
            "portfolio_metrics": {},
            "timestamp": datetime.now().isoformat()
        }