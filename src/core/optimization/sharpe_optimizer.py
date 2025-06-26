"""
Maximum Sharpe Ratio Optimization for Investment MCP System.

This module implements portfolio optimization to maximize the Sharpe ratio
with Swedish market constraints and considerations.
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


class SharpeOptimizer:
    """Maximum Sharpe ratio portfolio optimization implementation."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.constraints = PortfolioConstraints()
        
        # Swedish market parameters
        self.risk_free_rate = 0.02  # Swedish 10-year government bond approximation
        self.trading_costs = 0.0025  # 0.25% trading costs
        self.max_iterations = 1500
        self.tolerance = 1e-10
    
    def optimize_portfolio(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_tolerance: str,
        risk_free_rate: Optional[float] = None,
        current_portfolio: Optional[Dict[str, float]] = None,
        custom_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform maximum Sharpe ratio optimization.
        
        Args:
            expected_returns: Expected annual returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            risk_tolerance: Risk tolerance level (low/medium/high/very_high)
            risk_free_rate: Risk-free rate (if None, uses default)
            current_portfolio: Current portfolio weights for turnover constraints
            custom_constraints: Additional custom constraints
            
        Returns:
            Dictionary containing optimization results
        """
        
        try:
            # Use provided risk-free rate or default
            rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
            
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
            
            # Perform Sharpe ratio optimization
            result = self._maximize_sharpe_ratio(
                returns, cov_matrix, rf_rate, risk_tolerance,
                current_portfolio, custom_constraints
            )
            
            if result["success"]:
                # Create portfolio allocation
                weights = result["weights"]
                allocation = dict(zip(available_funds, weights))
                
                # Calculate portfolio metrics
                portfolio_metrics = self._calculate_portfolio_metrics(
                    allocation, returns, cov_matrix, rf_rate
                )
                
                # Add Swedish-specific analysis
                swedish_analysis = self._analyze_swedish_context(allocation)
                
                # Efficient frontier analysis
                frontier_analysis = self._analyze_efficient_frontier_position(
                    allocation, returns, cov_matrix, rf_rate
                )
                
                return {
                    "success": True,
                    "optimization_type": "maximum_sharpe_ratio",
                    "allocation": allocation,
                    "portfolio_metrics": portfolio_metrics,
                    "swedish_analysis": swedish_analysis,
                    "frontier_analysis": frontier_analysis,
                    "optimization_details": {
                        "risk_free_rate": rf_rate,
                        "risk_tolerance": risk_tolerance,
                        "n_iterations": result.get("n_iterations", 0),
                        "optimization_status": result.get("status", "unknown"),
                        "achieved_sharpe_ratio": result.get("sharpe_ratio", 0),
                        "objective_value": result.get("objective_value", 0)
                    },
                    "constraints_applied": self._get_applied_constraints(risk_tolerance),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return self._empty_result(result.get("error", "Sharpe optimization failed"))
                
        except Exception as e:
            self.logger.error(f"Sharpe ratio optimization failed: {e}")
            return self._empty_result(str(e))
    
    def _maximize_sharpe_ratio(
        self,
        returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float,
        risk_tolerance: str,
        current_portfolio: Optional[Dict[str, float]] = None,
        custom_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Maximize portfolio Sharpe ratio using multiple optimization approaches."""
        
        n_assets = len(returns)
        
        # Method 1: Direct Sharpe ratio maximization
        result_direct = self._optimize_sharpe_direct(
            returns, cov_matrix, risk_free_rate, risk_tolerance,
            current_portfolio, custom_constraints
        )
        
        # Method 2: Transform to quadratic problem (more stable)
        result_quad = self._optimize_sharpe_quadratic(
            returns, cov_matrix, risk_free_rate, risk_tolerance,
            current_portfolio, custom_constraints
        )
        
        # Choose best result
        best_result = result_direct
        if result_quad["success"] and (
            not result_direct["success"] or 
            result_quad.get("sharpe_ratio", 0) > result_direct.get("sharpe_ratio", 0)
        ):
            best_result = result_quad
        
        return best_result
    
    def _optimize_sharpe_direct(
        self,
        returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float,
        risk_tolerance: str,
        current_portfolio: Optional[Dict[str, float]] = None,
        custom_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Direct Sharpe ratio maximization."""
        
        n_assets = len(returns)
        
        # Objective function: negative Sharpe ratio (for minimization)
        def objective(weights):
            portfolio_return = np.dot(weights, returns.values)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix.values, weights))
            portfolio_std = np.sqrt(portfolio_variance)
            
            if portfolio_std <= 1e-12:  # Avoid division by zero
                return -1000  # High penalty
            
            excess_return = portfolio_return - risk_free_rate
            sharpe_ratio = excess_return / portfolio_std
            
            return -sharpe_ratio  # Negative for minimization
        
        # Constraints
        constraints = self._get_optimization_constraints(
            returns, cov_matrix, risk_tolerance, current_portfolio, custom_constraints
        )
        
        # Bounds
        bounds = self.constraints.get_weight_bounds(risk_tolerance, n_assets)
        
        # Try multiple starting points for robust optimization
        best_result = None
        best_sharpe = -np.inf
        
        starting_points = self._get_starting_points(returns, cov_matrix, n_assets)
        
        for start_point in starting_points:
            try:
                result = minimize(
                    objective,
                    start_point,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
                )
                
                if result.success:
                    # Calculate actual Sharpe ratio
                    weights = result.x
                    portfolio_return = np.dot(weights, returns.values)
                    portfolio_std = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))
                    actual_sharpe = (portfolio_return - risk_free_rate) / portfolio_std
                    
                    if actual_sharpe > best_sharpe:
                        best_sharpe = actual_sharpe
                        best_result = {
                            "success": True,
                            "weights": weights,
                            "sharpe_ratio": actual_sharpe,
                            "objective_value": result.fun,
                            "n_iterations": result.nit,
                            "status": result.message
                        }
                        
            except Exception as e:
                self.logger.warning(f"Direct Sharpe optimization attempt failed: {e}")
                continue
        
        if best_result:
            return best_result
        else:
            return {"success": False, "error": "All direct optimization attempts failed"}
    
    def _optimize_sharpe_quadratic(
        self,
        returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float,
        risk_tolerance: str,
        current_portfolio: Optional[Dict[str, float]] = None,
        custom_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Sharpe ratio optimization using quadratic transformation."""
        
        n_assets = len(returns)
        excess_returns = returns.values - risk_free_rate
        
        # Check if we have positive excess returns
        if np.all(excess_returns <= 0):
            return {"success": False, "error": "No assets with positive excess returns"}
        
        try:
            # Transform to auxiliary variable y = w / (w^T * mu)
            # Minimize y^T * Sigma * y subject to 1^T * y = 1 and excess_returns^T * y = 1
            
            def objective_quad(y):
                return np.dot(y, np.dot(cov_matrix.values, y))
            
            # Constraints for quadratic formulation
            constraints_quad = []
            
            # Sum constraint: excess_returns^T * y = 1
            constraints_quad.append({
                'type': 'eq',
                'fun': lambda y: np.dot(excess_returns, y) - 1.0
            })
            
            # Transform other constraints (this is simplified)
            # Full implementation would transform all original constraints
            
            # Bounds for y (can be negative)
            bounds_quad = [(-10, 10) for _ in range(n_assets)]
            
            # Initial guess
            y0 = np.ones(n_assets) / (n_assets * np.mean(excess_returns[excess_returns > 0]))
            
            result = minimize(
                objective_quad,
                y0,
                method='SLSQP',
                bounds=bounds_quad,
                constraints=constraints_quad,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            if result.success:
                y_opt = result.x
                # Transform back to weights
                weights = y_opt / np.sum(y_opt)
                
                # Ensure non-negative weights (project if needed)
                weights = np.maximum(weights, 0)
                weights = weights / np.sum(weights)  # Renormalize
                
                # Calculate Sharpe ratio
                portfolio_return = np.dot(weights, returns.values)
                portfolio_std = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
                
                return {
                    "success": True,
                    "weights": weights,
                    "sharpe_ratio": sharpe_ratio,
                    "objective_value": result.fun,
                    "n_iterations": result.nit,
                    "status": result.message
                }
            else:
                return {"success": False, "error": f"Quadratic optimization failed: {result.message}"}
                
        except Exception as e:
            return {"success": False, "error": f"Quadratic optimization error: {str(e)}"}
    
    def calculate_efficient_frontier(
        self,
        returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_tolerance: str,
        risk_free_rate: Optional[float] = None,
        n_points: int = 25
    ) -> Dict[str, Any]:
        """Calculate efficient frontier with tangency portfolio highlighted."""
        
        try:
            rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
            
            # First find the maximum Sharpe ratio portfolio (tangency portfolio)
            tangency_result = self.optimize_portfolio(
                returns, cov_matrix, risk_tolerance, rf_rate
            )
            
            if not tangency_result["success"]:
                return {"success": False, "error": "Could not find tangency portfolio"}
            
            tangency_allocation = tangency_result["allocation"]
            tangency_weights = np.array([tangency_allocation.get(fund, 0) for fund in returns.index])
            tangency_return = np.dot(tangency_weights, returns.values)
            tangency_vol = np.sqrt(np.dot(tangency_weights, np.dot(cov_matrix.values, tangency_weights)))
            tangency_sharpe = (tangency_return - rf_rate) / tangency_vol
            
            # Calculate efficient frontier points around the tangency portfolio
            min_return = returns.min()
            max_return = min(returns.max(), tangency_return * 1.5)  # Don't go too far beyond tangency
            
            target_returns = np.linspace(min_return, max_return, n_points)
            
            frontier_points = []
            
            for target_return in target_returns:
                # For each target return, minimize variance
                point_result = self._minimize_variance_for_return(
                    returns, cov_matrix, target_return, risk_tolerance
                )
                
                if point_result["success"]:
                    weights = point_result["weights"]
                    actual_return = np.dot(weights, returns.values)
                    volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))
                    sharpe = (actual_return - rf_rate) / volatility if volatility > 0 else 0
                    
                    frontier_points.append({
                        "target_return": float(target_return),
                        "return": float(actual_return),
                        "volatility": float(volatility),
                        "sharpe_ratio": float(sharpe),
                        "weights": weights.tolist()
                    })
            
            # Add capital allocation line (combinations of risk-free asset and tangency portfolio)
            cal_points = []
            for leverage in np.linspace(0, 2, 11):  # 0% to 200% in tangency portfolio
                cal_return = rf_rate + leverage * (tangency_return - rf_rate)
                cal_vol = leverage * tangency_vol
                cal_sharpe = (cal_return - rf_rate) / cal_vol if cal_vol > 0 else 0
                
                cal_points.append({
                    "leverage": float(leverage),
                    "return": float(cal_return),
                    "volatility": float(cal_vol),
                    "sharpe_ratio": float(cal_sharpe)
                })
            
            return {
                "success": True,
                "efficient_frontier": frontier_points,
                "tangency_portfolio": {
                    "allocation": tangency_allocation,
                    "return": float(tangency_return),
                    "volatility": float(tangency_vol),
                    "sharpe_ratio": float(tangency_sharpe)
                },
                "capital_allocation_line": cal_points,
                "risk_free_rate": rf_rate,
                "analysis": {
                    "max_sharpe_ratio": float(tangency_sharpe),
                    "efficient_frontier_points": len(frontier_points),
                    "risk_tolerance": risk_tolerance
                },
                "calculated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Efficient frontier calculation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _minimize_variance_for_return(
        self,
        returns: pd.Series,
        cov_matrix: pd.DataFrame,
        target_return: float,
        risk_tolerance: str
    ) -> Dict[str, Any]:
        """Minimize portfolio variance for a given target return."""
        
        n_assets = len(returns)
        
        # Objective function: minimize portfolio variance
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix.values, weights))
        
        # Constraints
        constraints = []
        
        # Return constraint
        constraints.append({
            'type': 'eq',
            'fun': lambda weights: np.dot(weights, returns.values) - target_return
        })
        
        # Weights sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda weights: np.sum(weights) - 1.0
        })
        
        # Add basic risk constraints (simplified for frontier calculation)
        max_weight = self.constraints.max_single_position.get(risk_tolerance, 0.35)
        for i in range(n_assets):
            constraints.append({
                'type': 'ineq',
                'fun': lambda weights, idx=i: max_weight - weights[idx]
            })
        
        # Bounds
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
        
        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets
        
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
                    "objective_value": result.fun
                }
            else:
                return {"success": False, "error": result.message}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_optimization_constraints(
        self,
        returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_tolerance: str,
        current_portfolio: Optional[Dict[str, float]] = None,
        custom_constraints: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get all constraints for Sharpe ratio optimization."""
        
        constraints = []
        fund_ids = returns.index.tolist()
        
        # Weights sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda weights: np.sum(weights) - 1.0
        })
        
        # Get comprehensive constraints from constraints module
        all_constraints = self.constraints.get_comprehensive_constraints(
            fund_ids=fund_ids,
            risk_tolerance=risk_tolerance,
            covariance_matrix=cov_matrix,
            expected_returns=returns,
            current_weights=None if current_portfolio is None else np.array([
                current_portfolio.get(fund, 0) for fund in fund_ids
            ]),
            custom_constraints=custom_constraints
        )
        
        constraints.extend(all_constraints)
        
        return constraints
    
    def _get_starting_points(
        self,
        returns: pd.Series,
        cov_matrix: pd.DataFrame,
        n_assets: int
    ) -> List[np.ndarray]:
        """Get multiple starting points for robust optimization."""
        
        starting_points = []
        
        # 1. Equal weights
        starting_points.append(np.ones(n_assets) / n_assets)
        
        # 2. Return-weighted (proportional to expected returns)
        positive_returns = np.maximum(returns.values, 0.001)
        return_weights = positive_returns / np.sum(positive_returns)
        starting_points.append(return_weights)
        
        # 3. Inverse volatility weighted
        volatilities = np.sqrt(np.diag(cov_matrix.values))
        inv_vol_weights = (1 / volatilities) / np.sum(1 / volatilities)
        starting_points.append(inv_vol_weights)
        
        # 4. Maximum return fund (corner solution)
        max_return_weights = np.zeros(n_assets)
        max_return_weights[np.argmax(returns.values)] = 1.0
        starting_points.append(max_return_weights)
        
        # 5. Minimum variance approximation
        try:
            inv_cov = np.linalg.inv(cov_matrix.values)
            ones = np.ones((n_assets, 1))
            min_var_weights = np.dot(inv_cov, ones) / np.dot(ones.T, np.dot(inv_cov, ones))
            min_var_weights = min_var_weights.flatten()
            min_var_weights = np.maximum(min_var_weights, 0)  # Ensure non-negative
            min_var_weights = min_var_weights / np.sum(min_var_weights)  # Normalize
            starting_points.append(min_var_weights)
        except np.linalg.LinAlgError:
            pass  # Skip if matrix is singular
        
        # 6. Random diversified portfolio
        np.random.seed(42)  # For reproducibility
        random_weights = np.random.dirichlet(np.ones(n_assets))
        starting_points.append(random_weights)
        
        return starting_points
    
    def _calculate_portfolio_metrics(
        self,
        allocation: Dict[str, float],
        returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float
    ) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics."""
        
        weights = np.array([allocation.get(fund, 0) for fund in returns.index])
        
        # Basic metrics
        portfolio_return = np.dot(weights, returns.values)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix.values, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Risk-adjusted metrics
        excess_return = portfolio_return - risk_free_rate
        sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Treynor ratio (simplified - assumes beta = 1 without market data)
        treynor_ratio = excess_return  # Would need market beta for proper calculation
        
        # Information ratio (relative to equal-weight benchmark)
        equal_weight_return = returns.mean()
        active_return = portfolio_return - equal_weight_return
        
        # Diversification metrics
        n_assets = len([w for w in weights if w > 0.01])  # Assets with >1% allocation
        effective_n_assets = 1 / np.sum(weights ** 2)  # Inverse of Herfindahl index
        concentration_ratio = max(weights)
        
        # Risk contribution analysis
        marginal_contribs = np.dot(cov_matrix.values, weights)
        risk_contribs = weights * marginal_contribs / portfolio_variance if portfolio_variance > 0 else weights
        
        return {
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "treynor_ratio": float(treynor_ratio),
            "active_return": float(active_return),
            "excess_return": float(excess_return),
            "number_of_assets": int(n_assets),
            "effective_number_of_assets": float(effective_n_assets),
            "concentration_ratio": float(concentration_ratio),
            "diversification_score": float(1 - concentration_ratio),
            "expected_annual_return_pct": float(portfolio_return * 100),
            "annual_volatility_pct": float(portfolio_volatility * 100),
            "risk_contributions": {
                fund: float(risk_contribs[i]) 
                for i, fund in enumerate(returns.index)
            }
        }
    
    def _analyze_efficient_frontier_position(
        self,
        allocation: Dict[str, float],
        returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float
    ) -> Dict[str, Any]:
        """Analyze portfolio's position on the efficient frontier."""
        
        weights = np.array([allocation.get(fund, 0) for fund in returns.index])
        portfolio_return = np.dot(weights, returns.values)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
        
        # Calculate if this is truly the maximum Sharpe ratio portfolio
        # by checking if the tangency condition holds approximately
        excess_returns = returns.values - risk_free_rate
        
        try:
            inv_cov = np.linalg.inv(cov_matrix.values)
            theoretical_weights = np.dot(inv_cov, excess_returns)
            theoretical_weights = theoretical_weights / np.sum(theoretical_weights)
            
            # Compare with actual weights
            weight_difference = np.linalg.norm(weights - theoretical_weights)
            is_tangency = weight_difference < 0.1  # Tolerance for approximation
            
        except np.linalg.LinAlgError:
            is_tangency = False
            theoretical_weights = None
            weight_difference = None
        
        return {
            "is_maximum_sharpe": is_tangency,
            "sharpe_ratio": float(sharpe_ratio),
            "portfolio_return": float(portfolio_return),
            "portfolio_volatility": float(portfolio_vol),
            "weight_difference_from_theoretical": float(weight_difference) if weight_difference is not None else None,
            "theoretical_weights": theoretical_weights.tolist() if theoretical_weights is not None else None,
            "frontier_efficiency": "Maximum Sharpe" if is_tangency else "Sub-optimal"
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
            elif "global" in category or any(region in fund_id.lower() for region in ["usa", "europa", "japan"]):
                categories["global_equity"] += weight
            elif category in ["cryptocurrency", "precious_metals"]:
                categories["alternatives"] += weight
            elif "real_estate" in category:
                categories["fixed_income"] += weight
        
        # Swedish investment considerations
        home_bias = categories["swedish_equity"] + categories["nordic_equity"]
        currency_diversification = 1 - home_bias
        
        # Tax efficiency
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
        
        # Global diversification
        if categories["global_equity"] < 0.3:
            recommendations.append("Öka global exponering för bättre diversifiering")
        
        # Alternative investments
        if categories["alternatives"] > 0.2:
            recommendations.append("Hög andel alternativa investeringar - överväg riskbegränsning")
        
        # Tax efficiency
        crypto_allocation = sum(
            weight for fund_id, weight in allocation.items()
            if any(crypto in fund_id.lower() for crypto in ["bitcoin", "ether"])
        )
        
        if crypto_allocation > 0.1:
            recommendations.append("Kryptovalutor passar bättre i AF-konto än ISK")
        
        # Sharpe ratio specific
        recommendations.append("Portföljen är optimerad för högsta riskjusterade avkastning")
        
        if not recommendations:
            recommendations.append("Portföljen är väl optimerad för svenska förhållanden")
        
        return recommendations
    
    def _get_applied_constraints(self, risk_tolerance: str) -> List[str]:
        """Get list of applied constraints for reporting."""
        
        constraints_list = [
            "Maximera Sharpe-kvot",
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
            "optimization_type": "maximum_sharpe_ratio",
            "allocation": {},
            "portfolio_metrics": {},
            "timestamp": datetime.now().isoformat()
        }