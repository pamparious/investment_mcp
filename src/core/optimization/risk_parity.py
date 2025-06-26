"""
Risk Parity Portfolio Optimization for Investment MCP System.

This module implements risk parity allocation where each asset contributes
equally to portfolio risk, with Swedish market constraints.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from scipy.optimize import minimize
from datetime import datetime

from ..config import TRADEABLE_FUNDS
from .constraints import PortfolioConstraints

logger = logging.getLogger(__name__)


class RiskParityOptimizer:
    """Risk parity portfolio optimization implementation."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.constraints = PortfolioConstraints()
        
        # Swedish market specific parameters
        self.risk_free_rate = 0.02
        self.trading_costs = 0.0025
        self.max_iterations = 1000
        self.tolerance = 1e-8
    
    def optimize_portfolio(
        self,
        covariance_matrix: pd.DataFrame,
        risk_tolerance: str,
        target_volatility: Optional[float] = None,
        current_portfolio: Optional[Dict[str, float]] = None,
        custom_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform risk parity optimization.
        
        Args:
            covariance_matrix: Covariance matrix of asset returns
            risk_tolerance: Risk tolerance level (low/medium/high/very_high)
            target_volatility: Target portfolio volatility (optional)
            current_portfolio: Current portfolio weights for turnover constraints
            custom_constraints: Additional custom constraints
            
        Returns:
            Dictionary containing optimization results
        """
        
        try:
            # Validate inputs
            if not self._validate_covariance_matrix(covariance_matrix):
                return self._empty_result("Invalid covariance matrix")
            
            # Filter to approved funds only
            approved_funds = list(TRADEABLE_FUNDS.keys())
            available_funds = [fund for fund in covariance_matrix.index if fund in approved_funds]
            
            if len(available_funds) < 3:
                return self._empty_result("Insufficient approved funds available")
            
            # Subset covariance matrix to available funds
            cov_matrix = covariance_matrix.loc[available_funds, available_funds]
            n_assets = len(available_funds)
            
            # Perform risk parity optimization
            result = self._optimize_risk_parity(
                cov_matrix, risk_tolerance, target_volatility,
                current_portfolio, custom_constraints
            )
            
            if result["success"]:
                # Create portfolio allocation
                weights = result["weights"]
                allocation = dict(zip(available_funds, weights))
                
                # Calculate portfolio metrics
                portfolio_metrics = self._calculate_portfolio_metrics(
                    allocation, cov_matrix
                )
                
                # Add Swedish-specific analysis
                swedish_analysis = self._analyze_swedish_context(allocation)
                
                return {
                    "success": True,
                    "optimization_type": "risk_parity",
                    "allocation": allocation,
                    "portfolio_metrics": portfolio_metrics,
                    "swedish_analysis": swedish_analysis,
                    "optimization_details": {
                        "target_volatility": target_volatility,
                        "risk_tolerance": risk_tolerance,
                        "n_iterations": result.get("n_iterations", 0),
                        "optimization_status": result.get("status", "unknown"),
                        "risk_concentration": result.get("risk_concentration", 0),
                        "equal_risk_achieved": result.get("equal_risk_achieved", False)
                    },
                    "constraints_applied": self._get_applied_constraints(risk_tolerance),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return self._empty_result(result.get("error", "Risk parity optimization failed"))
                
        except Exception as e:
            self.logger.error(f"Risk parity optimization failed: {e}")
            return self._empty_result(str(e))
    
    def _optimize_risk_parity(
        self,
        cov_matrix: pd.DataFrame,
        risk_tolerance: str,
        target_volatility: Optional[float] = None,
        current_portfolio: Optional[Dict[str, float]] = None,
        custom_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Optimize for equal risk contribution."""
        
        n_assets = len(cov_matrix)
        
        # Objective function: minimize sum of squared differences in risk contributions
        def objective(weights):
            # Calculate risk contributions
            portfolio_variance = np.dot(weights, np.dot(cov_matrix.values, weights))
            
            if portfolio_variance <= 1e-10:
                return 1e10  # High penalty for degenerate portfolios
            
            # Marginal contributions to risk
            marginal_contribs = np.dot(cov_matrix.values, weights)
            risk_contribs = weights * marginal_contribs / portfolio_variance
            
            # Target equal risk contribution (1/n for each asset)
            target_risk_contrib = 1.0 / n_assets
            
            # Sum of squared deviations from equal risk contribution
            risk_deviations = (risk_contribs - target_risk_contrib) ** 2
            
            return np.sum(risk_deviations)
        
        # Constraints
        constraints = []
        
        # Weights sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda weights: np.sum(weights) - 1.0
        })
        
        # Target volatility constraint (if specified)
        if target_volatility is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda weights: np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights))) - target_volatility
            })
        
        # Add risk-based constraints
        risk_constraints = self.constraints.get_risk_constraints(risk_tolerance, n_assets)
        constraints.extend(risk_constraints)
        
        # Add custom constraints
        if custom_constraints:
            constraints.extend(custom_constraints.get("constraints", []))
        
        # Bounds
        bounds = self.constraints.get_weight_bounds(risk_tolerance, n_assets)
        
        # Add turnover constraints if current portfolio provided
        if current_portfolio:
            current_weights = np.array([
                current_portfolio.get(fund, 0) for fund in cov_matrix.index
            ])
            turnover_constraint = self._get_turnover_constraint(
                current_weights, risk_tolerance
            )
            if turnover_constraint:
                constraints.append(turnover_constraint)
        
        # Try multiple starting points for robust optimization
        best_result = None
        best_objective = np.inf
        
        starting_points = [
            np.array([1.0 / n_assets] * n_assets),  # Equal weights
            self._get_inverse_volatility_weights(cov_matrix),  # Inverse volatility
            self._get_minimum_variance_weights(cov_matrix),  # Minimum variance starting point
        ]
        
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
                
                if result.success and result.fun < best_objective:
                    best_objective = result.fun
                    best_result = result
                    
            except Exception as e:
                self.logger.warning(f"Risk parity optimization attempt failed: {e}")
                continue
        
        if best_result and best_result.success:
            # Validate risk parity achievement
            weights = best_result.x
            risk_concentration = self._calculate_risk_concentration(weights, cov_matrix.values)
            equal_risk_achieved = risk_concentration < 0.1  # Less than 10% concentration
            
            return {
                "success": True,
                "weights": weights,
                "objective_value": best_result.fun,
                "n_iterations": best_result.nit,
                "status": best_result.message,
                "risk_concentration": risk_concentration,
                "equal_risk_achieved": equal_risk_achieved
            }
        else:
            return {
                "success": False,
                "error": "All risk parity optimization attempts failed"
            }
    
    def calculate_risk_contributions(
        self,
        weights: np.ndarray,
        cov_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate detailed risk contributions for a portfolio."""
        
        try:
            portfolio_variance = np.dot(weights, np.dot(cov_matrix.values, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            if portfolio_variance <= 1e-10:
                return {"error": "Degenerate portfolio variance"}
            
            # Marginal contributions to risk
            marginal_contribs = np.dot(cov_matrix.values, weights)
            
            # Component contributions to risk (percentage)
            risk_contribs = weights * marginal_contribs / portfolio_variance
            
            # Component contributions to volatility (absolute)
            vol_contribs = weights * marginal_contribs / portfolio_volatility
            
            # Create detailed breakdown
            fund_names = cov_matrix.index.tolist()
            risk_breakdown = {}
            
            for i, fund in enumerate(fund_names):
                risk_breakdown[fund] = {
                    "weight": float(weights[i]),
                    "risk_contribution_pct": float(risk_contribs[i] * 100),
                    "volatility_contribution": float(vol_contribs[i]),
                    "marginal_risk": float(marginal_contribs[i])
                }
            
            # Calculate risk concentration metrics
            risk_concentration = self._calculate_risk_concentration(weights, cov_matrix.values)
            effective_number_assets = 1 / np.sum(risk_contribs ** 2)
            
            return {
                "success": True,
                "portfolio_volatility": float(portfolio_volatility),
                "risk_breakdown": risk_breakdown,
                "risk_concentration": float(risk_concentration),
                "effective_number_of_assets": float(effective_number_assets),
                "equal_risk_target": 1.0 / len(weights),
                "risk_parity_achieved": risk_concentration < 0.1,
                "calculated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Risk contribution calculation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def create_volatility_targeted_portfolio(
        self,
        covariance_matrix: pd.DataFrame,
        target_volatility: float,
        risk_tolerance: str
    ) -> Dict[str, Any]:
        """Create risk parity portfolio with specific volatility target."""
        
        try:
            # First, get base risk parity weights
            base_result = self.optimize_portfolio(
                covariance_matrix, risk_tolerance, target_volatility
            )
            
            if not base_result["success"]:
                return base_result
            
            allocation = base_result["allocation"]
            weights = np.array([allocation[fund] for fund in covariance_matrix.index])
            
            # Calculate current portfolio volatility
            current_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix.values, weights)))
            
            # Scale weights to achieve target volatility
            if current_vol > 1e-10:
                scaling_factor = target_volatility / current_vol
                
                # If scaling factor is too high, we need to add cash/risk-free asset
                if scaling_factor > 1.0:
                    # Use leverage constraint based on risk tolerance
                    max_leverage = self._get_max_leverage(risk_tolerance)
                    if scaling_factor > max_leverage:
                        scaling_factor = max_leverage
                
                scaled_weights = weights * scaling_factor
                cash_weight = 1.0 - np.sum(scaled_weights)
                
                # Update allocation
                scaled_allocation = {}
                for i, fund in enumerate(covariance_matrix.index):
                    if scaled_weights[i] > 0.001:  # Minimum 0.1% allocation
                        scaled_allocation[fund] = float(scaled_weights[i])
                
                if cash_weight > 0.001:
                    scaled_allocation["CASH"] = float(cash_weight)
                
                # Recalculate metrics
                portfolio_metrics = self._calculate_portfolio_metrics(
                    scaled_allocation, covariance_matrix
                )
                
                return {
                    "success": True,
                    "optimization_type": "risk_parity_volatility_targeted",
                    "allocation": scaled_allocation,
                    "portfolio_metrics": portfolio_metrics,
                    "target_volatility": target_volatility,
                    "achieved_volatility": float(target_volatility),
                    "scaling_factor": float(scaling_factor),
                    "cash_allocation": float(cash_weight),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return self._empty_result("Degenerate portfolio volatility")
                
        except Exception as e:
            self.logger.error(f"Volatility-targeted portfolio creation failed: {e}")
            return self._empty_result(str(e))
    
    def _calculate_risk_concentration(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Calculate Herfindahl index of risk concentration."""
        
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        
        if portfolio_variance <= 1e-10:
            return 1.0  # Maximum concentration
        
        marginal_contribs = np.dot(cov_matrix, weights)
        risk_contribs = weights * marginal_contribs / portfolio_variance
        
        # Herfindahl index of risk contributions
        return float(np.sum(risk_contribs ** 2))
    
    def _get_inverse_volatility_weights(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Get inverse volatility weighted starting point."""
        
        volatilities = np.sqrt(np.diag(cov_matrix.values))
        inv_vol = 1.0 / volatilities
        weights = inv_vol / np.sum(inv_vol)
        
        return weights
    
    def _get_minimum_variance_weights(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Get minimum variance portfolio as starting point."""
        
        try:
            n_assets = len(cov_matrix)
            
            # Simple minimum variance solution
            inv_cov = np.linalg.inv(cov_matrix.values)
            ones = np.ones((n_assets, 1))
            
            weights = np.dot(inv_cov, ones)
            weights = weights / np.sum(weights)
            
            return weights.flatten()
            
        except np.linalg.LinAlgError:
            # Fallback to equal weights if covariance matrix is singular
            n_assets = len(cov_matrix)
            return np.array([1.0 / n_assets] * n_assets)
    
    def _get_max_leverage(self, risk_tolerance: str) -> float:
        """Get maximum leverage based on risk tolerance."""
        
        leverage_limits = {
            "low": 1.0,      # No leverage for conservative investors
            "medium": 1.1,   # 10% leverage allowed
            "high": 1.25,    # 25% leverage allowed
            "very_high": 1.5 # 50% leverage allowed
        }
        
        return leverage_limits.get(risk_tolerance, 1.0)
    
    def _get_turnover_constraint(
        self, 
        current_weights: np.ndarray, 
        risk_tolerance: str
    ) -> Optional[Dict[str, Any]]:
        """Get turnover constraint to limit transaction costs."""
        
        max_turnover = {
            "low": 0.2,      # Very conservative turnover
            "medium": 0.4,   # Moderate turnover
            "high": 0.6,     # Higher turnover acceptable  
            "very_high": 1.0 # No turnover constraint
        }
        
        max_turn = max_turnover.get(risk_tolerance, 0.4)
        
        if max_turn >= 1.0:
            return None  # No constraint
        
        return {
            'type': 'ineq',
            'fun': lambda weights: max_turn - np.sum(np.abs(weights - current_weights))
        }
    
    def _calculate_portfolio_metrics(
        self,
        allocation: Dict[str, float],
        cov_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics."""
        
        # Filter out cash for volatility calculations
        fund_allocation = {k: v for k, v in allocation.items() if k != "CASH"}
        
        if not fund_allocation:
            return {"error": "No fund allocations"}
        
        weights = np.array([fund_allocation.get(fund, 0) for fund in cov_matrix.index])
        
        # Basic metrics
        portfolio_variance = np.dot(weights, np.dot(cov_matrix.values, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Risk concentration metrics
        risk_concentration = self._calculate_risk_concentration(weights, cov_matrix.values)
        effective_n_assets = 1 / risk_concentration if risk_concentration > 0 else len(weights)
        
        # Diversification metrics
        n_assets = len([w for w in weights if w > 0.01])
        max_weight = max(weights) if len(weights) > 0 else 0
        
        return {
            "volatility": float(portfolio_volatility),
            "risk_concentration": float(risk_concentration),
            "effective_number_of_assets": float(effective_n_assets),
            "number_of_assets": int(n_assets),
            "max_weight": float(max_weight),
            "diversification_score": float(1 - max_weight),
            "annual_volatility_pct": float(portfolio_volatility * 100),
            "risk_parity_achieved": risk_concentration < 0.15,
            "cash_allocation": float(allocation.get("CASH", 0))
        }
    
    def _analyze_swedish_context(self, allocation: Dict[str, float]) -> Dict[str, Any]:
        """Analyze portfolio from Swedish investment perspective."""
        
        # Categorize allocations (excluding cash)
        fund_allocation = {k: v for k, v in allocation.items() if k != "CASH"}
        
        categories = {
            "swedish_equity": 0,
            "nordic_equity": 0,
            "global_equity": 0,
            "alternatives": 0,
            "real_estate": 0
        }
        
        for fund_id, weight in fund_allocation.items():
            fund_info = TRADEABLE_FUNDS.get(fund_id, {})
            fund_type = fund_info.get("type", "unknown")
            
            if "swedish" in fund_type or "sverige" in fund_id.lower():
                categories["swedish_equity"] += weight
            elif "nordic" in fund_type or "norden" in fund_id.lower():
                categories["nordic_equity"] += weight
            elif any(region in fund_type for region in ["global", "usa", "europe", "japan"]):
                categories["global_equity"] += weight
            elif fund_type in ["cryptocurrency", "precious_metals"]:
                categories["alternatives"] += weight
            elif "real_estate" in fund_type:
                categories["real_estate"] += weight
        
        # Swedish investment considerations
        home_bias = categories["swedish_equity"] + categories["nordic_equity"]
        currency_diversification = 1 - home_bias
        
        # Tax efficiency (Swedish perspective)
        isk_suitable_funds = sum(
            weight for fund_id, weight in fund_allocation.items()
            if not any(crypto in fund_id.lower() for crypto in ["bitcoin", "ether"])
        )
        
        return {
            "category_breakdown": categories,
            "home_bias": float(home_bias),
            "currency_diversification": float(currency_diversification),
            "isk_suitability": float(isk_suitable_funds),
            "swedish_tax_efficiency": "high" if isk_suitable_funds > 0.8 else "medium",
            "risk_parity_suitability": "excellent",  # Risk parity is always tax-efficient
            "recommendations": self._get_swedish_risk_parity_recommendations(categories, allocation)
        }
    
    def _get_swedish_risk_parity_recommendations(
        self, 
        categories: Dict[str, float], 
        allocation: Dict[str, float]
    ) -> List[str]:
        """Generate Swedish-specific risk parity recommendations."""
        
        recommendations = []
        
        # Risk parity specific advice
        recommendations.append("Risk parity ger jämn riskfördelning mellan tillgångar")
        
        # Home bias check
        if categories["swedish_equity"] > 0.3:
            recommendations.append("Överväg att minska svenska aktier för bättre global diversifiering")
        
        # Currency diversification
        total_domestic = categories["swedish_equity"] + categories["nordic_equity"]
        if total_domestic > 0.5:
            recommendations.append("Öka global exponering för valutadiversifiering")
        
        # Alternative investments
        if categories["alternatives"] > 0.2:
            recommendations.append("Hög andel alternativa investeringar - risk parity hjälper balansera risken")
        
        # Cash allocation
        cash_weight = allocation.get("CASH", 0)
        if cash_weight > 0.1:
            recommendations.append(f"Kontantandel på {cash_weight*100:.1f}% minskar portföljens risk")
        
        if not recommendations:
            recommendations.append("Risk parity-portföljen är väl balanserad för svenska förhållanden")
        
        return recommendations
    
    def _get_applied_constraints(self, risk_tolerance: str) -> List[str]:
        """Get list of applied constraints for reporting."""
        
        constraints_list = [
            "Lika riskbidrag från alla tillgångar",
            "Weights sum to 1.0",
            "Only approved Swedish funds",
            f"Risk tolerance: {risk_tolerance}"
        ]
        
        bounds = self.constraints.get_weight_bounds(risk_tolerance, 1)[0]
        constraints_list.append(f"Weight bounds: {bounds[0]:.1%} - {bounds[1]:.1%}")
        
        return constraints_list
    
    def _validate_covariance_matrix(self, cov_matrix: pd.DataFrame) -> bool:
        """Validate covariance matrix inputs."""
        
        if cov_matrix.empty:
            return False
        
        # Check if matrix is square
        if cov_matrix.shape[0] != cov_matrix.shape[1]:
            return False
        
        # Check if indices match columns
        if not (cov_matrix.index == cov_matrix.columns).all():
            return False
        
        # Check for NaN or infinite values
        if cov_matrix.isna().any().any():
            return False
        
        if np.isinf(cov_matrix.values).any():
            return False
        
        # Check if matrix is positive semi-definite
        try:
            eigenvals = np.linalg.eigvals(cov_matrix.values)
            if np.any(eigenvals < -1e-8):  # Allow for small numerical errors
                return False
        except np.linalg.LinAlgError:
            return False
        
        return True
    
    def _empty_result(self, error_message: str) -> Dict[str, Any]:
        """Return empty result with error message."""
        
        return {
            "success": False,
            "error": error_message,
            "optimization_type": "risk_parity",
            "allocation": {},
            "portfolio_metrics": {},
            "timestamp": datetime.now().isoformat()
        }