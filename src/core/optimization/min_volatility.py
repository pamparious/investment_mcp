"""
Minimum Volatility Portfolio Optimization for Investment MCP System.

This module implements portfolio optimization to minimize volatility
while maintaining diversification and Swedish market considerations.
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


class MinimumVolatilityOptimizer:
    """Minimum volatility portfolio optimization implementation."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.constraints = PortfolioConstraints()
        
        # Conservative optimization parameters for minimum volatility
        self.risk_free_rate = 0.02
        self.trading_costs = 0.0025
        self.max_iterations = 1000
        self.tolerance = 1e-10
    
    def optimize_portfolio(
        self,
        covariance_matrix: pd.DataFrame,
        risk_tolerance: str,
        expected_returns: Optional[pd.Series] = None,
        min_expected_return: Optional[float] = None,
        current_portfolio: Optional[Dict[str, float]] = None,
        custom_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform minimum volatility optimization.
        
        Args:
            covariance_matrix: Covariance matrix of asset returns
            risk_tolerance: Risk tolerance level (low/medium/high/very_high)
            expected_returns: Expected returns (optional, for return constraints)
            min_expected_return: Minimum required expected return (optional)
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
            
            # Subset data to available funds
            cov_matrix = covariance_matrix.loc[available_funds, available_funds]
            
            # Handle expected returns if provided
            returns = None
            if expected_returns is not None:
                returns = expected_returns[available_funds] if len(expected_returns) > 0 else None
            
            # Perform minimum volatility optimization
            result = self._minimize_volatility(
                cov_matrix, risk_tolerance, returns, min_expected_return,
                current_portfolio, custom_constraints
            )
            
            if result["success"]:
                # Create portfolio allocation
                weights = result["weights"]
                allocation = dict(zip(available_funds, weights))
                
                # Calculate portfolio metrics
                portfolio_metrics = self._calculate_portfolio_metrics(
                    allocation, cov_matrix, returns
                )
                
                # Add Swedish-specific analysis
                swedish_analysis = self._analyze_swedish_context(allocation)
                
                # Risk analysis
                risk_analysis = self._analyze_risk_characteristics(allocation, cov_matrix)
                
                return {
                    "success": True,
                    "optimization_type": "minimum_volatility",
                    "allocation": allocation,
                    "portfolio_metrics": portfolio_metrics,
                    "swedish_analysis": swedish_analysis,
                    "risk_analysis": risk_analysis,
                    "optimization_details": {
                        "min_expected_return": min_expected_return,
                        "risk_tolerance": risk_tolerance,
                        "n_iterations": result.get("n_iterations", 0),
                        "optimization_status": result.get("status", "unknown"),
                        "achieved_volatility": result.get("volatility", 0),
                        "objective_value": result.get("objective_value", 0)
                    },
                    "constraints_applied": self._get_applied_constraints(risk_tolerance),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return self._empty_result(result.get("error", "Minimum volatility optimization failed"))
                
        except Exception as e:
            self.logger.error(f"Minimum volatility optimization failed: {e}")
            return self._empty_result(str(e))
    
    def _minimize_volatility(
        self,
        cov_matrix: pd.DataFrame,
        risk_tolerance: str,
        expected_returns: Optional[pd.Series] = None,
        min_expected_return: Optional[float] = None,
        current_portfolio: Optional[Dict[str, float]] = None,
        custom_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Minimize portfolio volatility subject to constraints."""
        
        n_assets = len(cov_matrix)
        
        # Objective function: minimize portfolio variance (volatility squared)
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix.values, weights))
        
        # Constraints
        constraints = self._get_optimization_constraints(
            cov_matrix, risk_tolerance, expected_returns, min_expected_return,
            current_portfolio, custom_constraints
        )
        
        # Bounds
        bounds = self.constraints.get_weight_bounds(risk_tolerance, n_assets)
        
        # Try multiple starting points for robust optimization
        best_result = None
        best_volatility = np.inf
        
        starting_points = self._get_starting_points(cov_matrix, expected_returns, n_assets)
        
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
                    # Calculate actual volatility
                    weights = result.x
                    volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))
                    
                    if volatility < best_volatility:
                        best_volatility = volatility
                        best_result = {
                            "success": True,
                            "weights": weights,
                            "volatility": volatility,
                            "objective_value": result.fun,
                            "n_iterations": result.nit,
                            "status": result.message
                        }
                        
            except Exception as e:
                self.logger.warning(f"Minimum volatility optimization attempt failed: {e}")
                continue
        
        if best_result:
            return best_result
        else:
            return {"success": False, "error": "All optimization attempts failed"}
    
    def create_targeted_volatility_portfolio(
        self,
        covariance_matrix: pd.DataFrame,
        target_volatility: float,
        risk_tolerance: str,
        expected_returns: Optional[pd.Series] = None,
        optimize_criterion: str = "return"
    ) -> Dict[str, Any]:
        """
        Create portfolio with specific target volatility.
        
        Args:
            covariance_matrix: Covariance matrix
            target_volatility: Target portfolio volatility
            risk_tolerance: Risk tolerance level
            expected_returns: Expected returns (required if optimizing for return)
            optimize_criterion: What to optimize ("return", "sharpe", or "equal_weight")
            
        Returns:
            Portfolio optimization result with target volatility
        """
        
        try:
            # Validate inputs
            if optimize_criterion == "return" and expected_returns is None:
                return self._empty_result("Expected returns required for return optimization")
            
            # Get approved funds
            approved_funds = list(TRADEABLE_FUNDS.keys())
            available_funds = [fund for fund in covariance_matrix.index if fund in approved_funds]
            
            if len(available_funds) < 3:
                return self._empty_result("Insufficient approved funds available")
            
            # Subset data
            cov_matrix = covariance_matrix.loc[available_funds, available_funds]
            returns = expected_returns[available_funds] if expected_returns is not None else None
            
            n_assets = len(available_funds)
            
            # Define objective based on criterion
            if optimize_criterion == "return" and returns is not None:
                # Maximize expected return subject to volatility constraint
                def objective(weights):
                    return -np.dot(weights, returns.values)  # Negative for maximization
            elif optimize_criterion == "sharpe" and returns is not None:
                # Maximize Sharpe ratio subject to volatility constraint
                def objective(weights):
                    portfolio_return = np.dot(weights, returns.values)
                    excess_return = portfolio_return - self.risk_free_rate
                    return -excess_return  # Negative for maximization, volatility is constrained
            else:
                # Minimize concentration (maximize diversification)
                def objective(weights):
                    return np.sum(weights ** 2)  # Minimize sum of squared weights
            
            # Constraints
            constraints = []
            
            # Weights sum to 1
            constraints.append({
                'type': 'eq',
                'fun': lambda weights: np.sum(weights) - 1.0
            })
            
            # Target volatility constraint
            constraints.append({
                'type': 'eq',
                'fun': lambda weights: np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights))) - target_volatility
            })
            
            # Add other constraints
            fund_ids = available_funds
            other_constraints = self.constraints.get_comprehensive_constraints(
                fund_ids=fund_ids,
                risk_tolerance=risk_tolerance,
                covariance_matrix=cov_matrix,
                expected_returns=returns
            )
            constraints.extend(other_constraints)
            
            # Bounds
            bounds = self.constraints.get_weight_bounds(risk_tolerance, n_assets)
            
            # Starting point
            initial_weights = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            if result.success:
                weights = result.x
                allocation = dict(zip(available_funds, weights))
                
                # Calculate metrics
                portfolio_metrics = self._calculate_portfolio_metrics(
                    allocation, cov_matrix, returns
                )
                
                # Verify target volatility achieved
                achieved_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))
                volatility_error = abs(achieved_volatility - target_volatility)
                
                return {
                    "success": True,
                    "optimization_type": f"targeted_volatility_{optimize_criterion}",
                    "allocation": allocation,
                    "portfolio_metrics": portfolio_metrics,
                    "target_volatility": float(target_volatility),
                    "achieved_volatility": float(achieved_volatility),
                    "volatility_error": float(volatility_error),
                    "optimization_criterion": optimize_criterion,
                    "optimization_details": {
                        "n_iterations": result.nit,
                        "status": result.message,
                        "objective_value": result.fun
                    },
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return self._empty_result(f"Targeted volatility optimization failed: {result.message}")
                
        except Exception as e:
            self.logger.error(f"Targeted volatility optimization failed: {e}")
            return self._empty_result(str(e))
    
    def analyze_volatility_contributions(
        self,
        allocation: Dict[str, float],
        covariance_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze how each asset contributes to portfolio volatility."""
        
        try:
            # Get weights for available funds
            available_funds = [fund for fund in allocation.keys() if fund in covariance_matrix.index]
            weights = np.array([allocation[fund] for fund in available_funds])
            cov_matrix = covariance_matrix.loc[available_funds, available_funds]
            
            # Portfolio variance and volatility
            portfolio_variance = np.dot(weights, np.dot(cov_matrix.values, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            if portfolio_variance <= 1e-12:
                return {"error": "Portfolio variance too small for analysis"}
            
            # Marginal contributions to volatility
            marginal_vol_contribs = np.dot(cov_matrix.values, weights) / portfolio_volatility
            
            # Component contributions to volatility
            vol_contribs = weights * marginal_vol_contribs
            
            # Percentage contributions to variance
            marginal_var_contribs = np.dot(cov_matrix.values, weights)
            var_contribs = weights * marginal_var_contribs / portfolio_variance
            
            # Create detailed breakdown
            contributions = {}
            for i, fund in enumerate(available_funds):
                fund_info = TRADEABLE_FUNDS.get(fund, {})
                individual_vol = np.sqrt(cov_matrix.iloc[i, i])
                
                contributions[fund] = {
                    "weight": float(weights[i]),
                    "individual_volatility": float(individual_vol),
                    "marginal_volatility_contribution": float(marginal_vol_contribs[i]),
                    "total_volatility_contribution": float(vol_contribs[i]),
                    "variance_contribution_pct": float(var_contribs[i] * 100),
                    "volatility_ratio": float(marginal_vol_contribs[i] / individual_vol) if individual_vol > 0 else 0,
                    "fund_category": fund_info.get("category", "unknown")
                }
            
            # Calculate diversification benefit
            weighted_avg_vol = np.sum(weights * np.sqrt(np.diag(cov_matrix.values)))
            diversification_ratio = portfolio_volatility / weighted_avg_vol if weighted_avg_vol > 0 else 1
            diversification_benefit = 1 - diversification_ratio
            
            # Correlation analysis
            correlation_matrix = cov_matrix.corr()
            avg_correlation = np.mean(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)])
            max_correlation = np.max(correlation_matrix.values - np.eye(len(correlation_matrix)))
            min_correlation = np.min(correlation_matrix.values - np.eye(len(correlation_matrix)))
            
            return {
                "success": True,
                "portfolio_volatility": float(portfolio_volatility),
                "portfolio_variance": float(portfolio_variance),
                "fund_contributions": contributions,
                "diversification_analysis": {
                    "weighted_average_volatility": float(weighted_avg_vol),
                    "diversification_ratio": float(diversification_ratio),
                    "diversification_benefit": float(diversification_benefit),
                    "diversification_benefit_pct": float(diversification_benefit * 100)
                },
                "correlation_analysis": {
                    "average_correlation": float(avg_correlation),
                    "maximum_correlation": float(max_correlation),
                    "minimum_correlation": float(min_correlation),
                    "correlation_matrix": correlation_matrix.to_dict()
                },
                "risk_concentration": {
                    "effective_number_of_assets": float(1 / np.sum(var_contribs ** 2)),
                    "concentration_ratio": float(np.max(var_contribs)),
                    "top_3_risk_contributors": sorted(
                        [(fund, contributions[fund]["variance_contribution_pct"]) 
                         for fund in contributions.keys()],
                        key=lambda x: x[1], reverse=True
                    )[:3]
                },
                "calculated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Volatility contribution analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_optimization_constraints(
        self,
        cov_matrix: pd.DataFrame,
        risk_tolerance: str,
        expected_returns: Optional[pd.Series] = None,
        min_expected_return: Optional[float] = None,
        current_portfolio: Optional[Dict[str, float]] = None,
        custom_constraints: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get all constraints for minimum volatility optimization."""
        
        constraints = []
        fund_ids = cov_matrix.index.tolist()
        
        # Weights sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda weights: np.sum(weights) - 1.0
        })
        
        # Minimum expected return constraint
        if expected_returns is not None and min_expected_return is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda weights: np.dot(weights, expected_returns.values) - min_expected_return
            })
        
        # Get comprehensive constraints from constraints module
        all_constraints = self.constraints.get_comprehensive_constraints(
            fund_ids=fund_ids,
            risk_tolerance=risk_tolerance,
            covariance_matrix=cov_matrix,
            expected_returns=expected_returns,
            current_weights=None if current_portfolio is None else np.array([
                current_portfolio.get(fund, 0) for fund in fund_ids
            ]),
            custom_constraints=custom_constraints
        )
        
        constraints.extend(all_constraints)
        
        return constraints
    
    def _get_starting_points(
        self,
        cov_matrix: pd.DataFrame,
        expected_returns: Optional[pd.Series],
        n_assets: int
    ) -> List[np.ndarray]:
        """Get multiple starting points for robust optimization."""
        
        starting_points = []
        
        # 1. Equal weights
        starting_points.append(np.ones(n_assets) / n_assets)
        
        # 2. Inverse volatility weighted
        volatilities = np.sqrt(np.diag(cov_matrix.values))
        inv_vol_weights = (1 / volatilities) / np.sum(1 / volatilities)
        starting_points.append(inv_vol_weights)
        
        # 3. Analytical minimum variance solution (if possible)
        try:
            inv_cov = np.linalg.inv(cov_matrix.values)
            ones = np.ones((n_assets, 1))
            min_var_weights = np.dot(inv_cov, ones) / np.dot(ones.T, np.dot(inv_cov, ones))
            min_var_weights = min_var_weights.flatten()
            
            # Ensure non-negative and normalized
            min_var_weights = np.maximum(min_var_weights, 0)
            if np.sum(min_var_weights) > 0:
                min_var_weights = min_var_weights / np.sum(min_var_weights)
                starting_points.append(min_var_weights)
        except np.linalg.LinAlgError:
            # Skip if matrix is singular
            pass
        
        # 4. Low correlation assets (if we have correlation data)
        try:
            corr_matrix = cov_matrix.corr()
            avg_correlations = corr_matrix.mean()
            # Weight inversely to average correlation
            low_corr_weights = (1 / (1 + avg_correlations))
            low_corr_weights = low_corr_weights / np.sum(low_corr_weights)
            starting_points.append(low_corr_weights.values)
        except:
            pass
        
        # 5. Random diversified portfolio
        np.random.seed(42)  # For reproducibility
        random_weights = np.random.dirichlet(np.ones(n_assets))
        starting_points.append(random_weights)
        
        # 6. Conservative allocation (if we know fund types)
        conservative_weights = np.ones(n_assets) / n_assets
        fund_ids = cov_matrix.index.tolist()
        
        for i, fund_id in enumerate(fund_ids):
            fund_info = TRADEABLE_FUNDS.get(fund_id, {})
            risk_level = fund_info.get("risk_level", "medium")
            
            # Weight lower-risk funds more heavily
            if risk_level == "low":
                conservative_weights[i] *= 2.0
            elif risk_level == "very_high":
                conservative_weights[i] *= 0.5
        
        conservative_weights = conservative_weights / np.sum(conservative_weights)
        starting_points.append(conservative_weights)
        
        return starting_points
    
    def _calculate_portfolio_metrics(
        self,
        allocation: Dict[str, float],
        cov_matrix: pd.DataFrame,
        expected_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics."""
        
        weights = np.array([allocation.get(fund, 0) for fund in cov_matrix.index])
        
        # Basic risk metrics
        portfolio_variance = np.dot(weights, np.dot(cov_matrix.values, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Expected return (if available)
        portfolio_return = None
        sharpe_ratio = None
        if expected_returns is not None:
            portfolio_return = np.dot(weights, expected_returns.values)
            excess_return = portfolio_return - self.risk_free_rate
            sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Diversification metrics
        n_assets = len([w for w in weights if w > 0.01])  # Assets with >1% allocation
        effective_n_assets = 1 / np.sum(weights ** 2)  # Inverse of Herfindahl index
        concentration_ratio = max(weights)
        
        # Risk decomposition
        marginal_contribs = np.dot(cov_matrix.values, weights)
        risk_contribs = weights * marginal_contribs / portfolio_variance if portfolio_variance > 0 else weights
        max_risk_contrib = max(risk_contribs)
        
        metrics = {
            "volatility": float(portfolio_volatility),
            "variance": float(portfolio_variance),
            "number_of_assets": int(n_assets),
            "effective_number_of_assets": float(effective_n_assets),
            "concentration_ratio": float(concentration_ratio),
            "diversification_score": float(1 - concentration_ratio),
            "annual_volatility_pct": float(portfolio_volatility * 100),
            "max_risk_contribution": float(max_risk_contrib),
            "risk_concentration": float(max_risk_contrib)
        }
        
        # Add return-based metrics if available
        if portfolio_return is not None:
            metrics.update({
                "expected_return": float(portfolio_return),
                "expected_annual_return_pct": float(portfolio_return * 100),
                "sharpe_ratio": float(sharpe_ratio) if sharpe_ratio is not None else None,
                "return_to_risk_ratio": float(portfolio_return / portfolio_volatility) if portfolio_volatility > 0 else None
            })
        
        return metrics
    
    def _analyze_risk_characteristics(
        self,
        allocation: Dict[str, float],
        cov_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze risk characteristics specific to minimum volatility portfolios."""
        
        weights = np.array([allocation.get(fund, 0) for fund in cov_matrix.index])
        fund_ids = cov_matrix.index.tolist()
        
        # Correlation structure analysis
        corr_matrix = cov_matrix.corr()
        
        # Portfolio correlation characteristics
        portfolio_correlations = []
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                corr = corr_matrix.iloc[i, j]
                weight_product = weights[i] * weights[j]
                portfolio_correlations.append({
                    "fund_pair": (fund_ids[i], fund_ids[j]),
                    "correlation": float(corr),
                    "weight_product": float(weight_product),
                    "correlation_contribution": float(corr * weight_product)
                })
        
        # Effective correlation
        total_weight_products = sum(item["weight_product"] for item in portfolio_correlations)
        effective_correlation = sum(
            item["correlation_contribution"] for item in portfolio_correlations
        ) / total_weight_products if total_weight_products > 0 else 0
        
        # Volatility decomposition
        individual_volatilities = np.sqrt(np.diag(cov_matrix.values))
        weighted_avg_vol = np.sum(weights * individual_volatilities)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix.values, weights)))
        
        # Diversification benefit
        diversification_ratio = portfolio_vol / weighted_avg_vol if weighted_avg_vol > 0 else 1
        
        # Risk budget analysis
        marginal_contribs = np.dot(cov_matrix.values, weights)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix.values, weights))
        risk_contribs = weights * marginal_contribs / portfolio_variance if portfolio_variance > 0 else weights
        
        # Identify risk concentrations
        risk_budget = {}
        for i, fund_id in enumerate(fund_ids):
            fund_info = TRADEABLE_FUNDS.get(fund_id, {})
            risk_budget[fund_id] = {
                "weight": float(weights[i]),
                "risk_contribution": float(risk_contribs[i]),
                "individual_volatility": float(individual_volatilities[i]),
                "fund_category": fund_info.get("category", "unknown"),
                "risk_multiplier": float(risk_contribs[i] / weights[i]) if weights[i] > 0 else 0
            }
        
        return {
            "effective_correlation": float(effective_correlation),
            "diversification_ratio": float(diversification_ratio),
            "diversification_benefit": float(1 - diversification_ratio),
            "weighted_average_volatility": float(weighted_avg_vol),
            "portfolio_volatility": float(portfolio_vol),
            "volatility_reduction": float(weighted_avg_vol - portfolio_vol),
            "risk_budget": risk_budget,
            "correlation_analysis": {
                "average_correlation": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()),
                "max_correlation": float(corr_matrix.values.max() - 1),  # Subtract 1 to exclude diagonal
                "min_correlation": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min()),
                "portfolio_correlations": portfolio_correlations[:10]  # Top 10 by weight product
            },
            "concentration_metrics": {
                "max_weight": float(max(weights)),
                "max_risk_contribution": float(max(risk_contribs)),
                "effective_number_of_assets": float(1 / np.sum(weights ** 2)),
                "risk_effective_assets": float(1 / np.sum(risk_contribs ** 2))
            }
        }
    
    def _analyze_swedish_context(self, allocation: Dict[str, float]) -> Dict[str, Any]:
        """Analyze portfolio from Swedish conservative investment perspective."""
        
        # Categorize allocations
        categories = {
            "swedish_equity": 0,
            "nordic_equity": 0,
            "global_equity": 0,
            "alternatives": 0,
            "defensive": 0
        }
        
        for fund_id, weight in allocation.items():
            fund_info = TRADEABLE_FUNDS.get(fund_id, {})
            category = fund_info.get("category", "unknown")
            risk_level = fund_info.get("risk_level", "medium")
            
            if "swedish" in category or "sverige" in fund_id.lower():
                categories["swedish_equity"] += weight
            elif "nordic" in category or "norden" in fund_id.lower():
                categories["nordic_equity"] += weight
            elif "global" in category or any(region in fund_id.lower() for region in ["usa", "europa", "japan"]):
                categories["global_equity"] += weight
            elif category in ["cryptocurrency", "precious_metals"]:
                categories["alternatives"] += weight
            elif risk_level in ["low"] or "real_estate" in category:
                categories["defensive"] += weight
        
        # Conservative investing considerations
        home_bias = categories["swedish_equity"] + categories["nordic_equity"]
        defensive_allocation = categories["defensive"]
        
        # Tax efficiency for conservative investors
        isk_suitable = sum(
            weight for fund_id, weight in allocation.items()
            if not any(crypto in fund_id.lower() for crypto in ["bitcoin", "ether"])
        )
        
        # Conservative suitability score
        conservative_score = 0
        if defensive_allocation > 0.2:
            conservative_score += 0.3
        if categories["alternatives"] < 0.1:
            conservative_score += 0.3
        if home_bias < 0.6:  # Some international diversification
            conservative_score += 0.2
        if len([w for w in allocation.values() if w > 0.05]) >= 5:  # Good diversification
            conservative_score += 0.2
        
        return {
            "category_breakdown": categories,
            "home_bias": float(home_bias),
            "defensive_allocation": float(defensive_allocation),
            "alternative_allocation": float(categories["alternatives"]),
            "international_diversification": float(1 - home_bias),
            "isk_suitability": float(isk_suitable),
            "conservative_suitability_score": float(conservative_score),
            "conservative_rating": "High" if conservative_score > 0.8 else "Medium" if conservative_score > 0.5 else "Low",
            "recommendations": self._get_conservative_recommendations(categories, allocation)
        }
    
    def _get_conservative_recommendations(
        self,
        categories: Dict[str, float],
        allocation: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations for conservative minimum volatility portfolios."""
        
        recommendations = []
        
        # Conservative-specific advice
        recommendations.append("Minimumvolatilitet ger lägsta möjliga risk för portföljen")
        
        # Defensive allocation
        if categories["defensive"] < 0.2:
            recommendations.append("Överväg att öka defensiva tillgångar som fastigheter")
        
        # Alternatives warning
        if categories["alternatives"] > 0.15:
            recommendations.append("Hög andel alternativa tillgångar kan öka volatiliteten")
        
        # Diversification
        significant_positions = len([w for w in allocation.values() if w > 0.05])
        if significant_positions < 5:
            recommendations.append("Överväg fler positioner för bättre diversifiering")
        
        # Swedish context
        if categories["swedish_equity"] > 0.4:
            recommendations.append("Hög svenska andel - överväg mer global diversifiering")
        
        # Tax efficiency
        recommendations.append("Idealisk för ISK-konto med låg omsättning")
        
        return recommendations
    
    def _get_applied_constraints(self, risk_tolerance: str) -> List[str]:
        """Get list of applied constraints for reporting."""
        
        constraints_list = [
            "Minimera portföljvolatilitet",
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
            "optimization_type": "minimum_volatility",
            "allocation": {},
            "portfolio_metrics": {},
            "timestamp": datetime.now().isoformat()
        }