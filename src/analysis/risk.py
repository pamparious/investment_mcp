"""
Unified risk analysis module for Investment MCP System.

This module consolidates all risk analysis functionality from various scattered
analysis modules into a single, comprehensive risk assessment system.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


class RiskAnalyzer:
    """Comprehensive risk analysis for funds and portfolios."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        # Suppress warnings for cleaner output
        warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    def calculate_basic_risk_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Calculate basic risk metrics for a return series.
        
        Args:
            returns: Series of daily returns
            
        Returns:
            Dictionary containing basic risk metrics
        """
        
        if returns.empty or len(returns) < 2:
            return self._empty_risk_metrics()
        
        # Remove infinite and NaN values
        returns_clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(returns_clean) < 2:
            return self._empty_risk_metrics()
        
        try:
            # Basic statistics
            mean_return = returns_clean.mean()
            std_return = returns_clean.std()
            
            # Annualized metrics (assuming daily returns)
            trading_days = 252
            annualized_return = mean_return * trading_days
            annualized_volatility = std_return * np.sqrt(trading_days)
            
            # Sharpe ratio (assuming 0% risk-free rate for simplicity)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
            
            # Downside metrics
            negative_returns = returns_clean[returns_clean < 0]
            downside_deviation = negative_returns.std() * np.sqrt(trading_days) if len(negative_returns) > 0 else 0
            
            # Sortino ratio
            sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
            
            # Skewness and kurtosis
            skewness = stats.skew(returns_clean)
            kurtosis = stats.kurtosis(returns_clean)
            
            # Value at Risk (VaR)
            var_95 = np.percentile(returns_clean, 5)
            var_99 = np.percentile(returns_clean, 1)
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = returns_clean[returns_clean <= var_95].mean()
            cvar_99 = returns_clean[returns_clean <= var_99].mean()
            
            # Maximum drawdown
            max_drawdown = self._calculate_max_drawdown(returns_clean)
            
            return {
                "mean_return": float(mean_return),
                "std_return": float(std_return),
                "annualized_return": float(annualized_return),
                "annualized_volatility": float(annualized_volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "sortino_ratio": float(sortino_ratio),
                "downside_deviation": float(downside_deviation),
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "var_95": float(var_95),
                "var_99": float(var_99),
                "cvar_95": float(cvar_95),
                "cvar_99": float(cvar_99),
                "max_drawdown": float(max_drawdown),
                "positive_days": int((returns_clean > 0).sum()),
                "negative_days": int((returns_clean < 0).sum()),
                "total_days": int(len(returns_clean)),
                "win_rate": float((returns_clean > 0).mean())
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return self._empty_risk_metrics()
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        
        if returns.empty:
            return 0.0
        
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cum_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cum_returns - running_max) / running_max
        
        return float(drawdown.min())
    
    def calculate_portfolio_risk(
        self, 
        returns_matrix: pd.DataFrame, 
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate portfolio risk metrics given asset returns and weights.
        
        Args:
            returns_matrix: DataFrame with asset returns (columns = assets, rows = dates)
            weights: Dictionary mapping asset names to portfolio weights
            
        Returns:
            Dictionary containing portfolio risk metrics
        """
        
        if returns_matrix.empty or not weights:
            return self._empty_portfolio_risk_metrics()
        
        try:
            # Align weights with returns matrix columns
            weight_vector = []
            aligned_returns = pd.DataFrame()
            
            for asset, weight in weights.items():
                if asset in returns_matrix.columns:
                    weight_vector.append(weight)
                    aligned_returns[asset] = returns_matrix[asset]
                else:
                    self.logger.warning(f"Asset {asset} not found in returns matrix")
            
            if not weight_vector or aligned_returns.empty:
                return self._empty_portfolio_risk_metrics()
            
            # Convert to numpy arrays
            weight_vector = np.array(weight_vector)
            returns_array = aligned_returns.dropna().values
            
            if len(returns_array) < 2:
                return self._empty_portfolio_risk_metrics()
            
            # Calculate portfolio returns
            portfolio_returns = np.dot(returns_array, weight_vector)
            portfolio_returns_series = pd.Series(portfolio_returns)
            
            # Calculate portfolio risk metrics
            portfolio_metrics = self.calculate_basic_risk_metrics(portfolio_returns_series)
            
            # Calculate correlation matrix
            correlation_matrix = aligned_returns.corr()
            
            # Calculate portfolio volatility using covariance matrix
            cov_matrix = aligned_returns.cov()
            portfolio_variance = np.dot(weight_vector, np.dot(cov_matrix.values, weight_vector))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Annualized portfolio volatility
            annualized_portfolio_volatility = portfolio_volatility * np.sqrt(252)
            
            # Diversification ratio
            individual_volatilities = aligned_returns.std().values
            weighted_avg_volatility = np.dot(weight_vector, individual_volatilities)
            diversification_ratio = weighted_avg_volatility / portfolio_volatility if portfolio_volatility > 0 else 1
            
            # Add portfolio-specific metrics
            portfolio_metrics.update({
                "portfolio_volatility": float(portfolio_volatility),
                "annualized_portfolio_volatility": float(annualized_portfolio_volatility),
                "diversification_ratio": float(diversification_ratio),
                "number_of_assets": len(weight_vector),
                "effective_number_of_assets": float(1 / np.sum(weight_vector ** 2)),
                "concentration_ratio": float(max(weight_vector)),
                "correlation_matrix": correlation_matrix.to_dict(),
                "weights": weights
            })
            
            return portfolio_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk: {e}")
            return self._empty_portfolio_risk_metrics()
    
    def calculate_fund_correlation_matrix(self, returns_matrix: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix for funds."""
        
        if returns_matrix.empty:
            return {}
        
        try:
            correlation_matrix = returns_matrix.corr()
            return correlation_matrix.to_dict()
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {e}")
            return {}
    
    def identify_risk_factors(self, returns_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Identify key risk factors affecting the portfolio."""
        
        if returns_matrix.empty or len(returns_matrix.columns) < 2:
            return {}
        
        try:
            # Calculate correlation matrix
            corr_matrix = returns_matrix.corr()
            
            # Find highly correlated pairs (>0.7)
            high_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    asset1 = corr_matrix.columns[i]
                    asset2 = corr_matrix.columns[j]
                    correlation = corr_matrix.iloc[i, j]
                    
                    if abs(correlation) > 0.7:
                        high_correlations.append({
                            "asset1": asset1,
                            "asset2": asset2,
                            "correlation": float(correlation)
                        })
            
            # Calculate volatility ranking
            volatilities = returns_matrix.std().sort_values(ascending=False)
            
            # Identify assets with highest volatility
            high_volatility_assets = volatilities.head(3).to_dict()
            
            # Calculate beta relative to equal-weighted portfolio
            equal_weight_portfolio = returns_matrix.mean(axis=1)
            betas = {}
            
            for asset in returns_matrix.columns:
                asset_returns = returns_matrix[asset].dropna()
                portfolio_returns = equal_weight_portfolio[asset_returns.index]
                
                if len(asset_returns) > 1 and len(portfolio_returns) > 1:
                    covariance = np.cov(asset_returns, portfolio_returns)[0, 1]
                    portfolio_variance = np.var(portfolio_returns)
                    beta = covariance / portfolio_variance if portfolio_variance > 0 else 1
                    betas[asset] = float(beta)
            
            return {
                "high_correlations": high_correlations,
                "high_volatility_assets": {k: float(v) for k, v in high_volatility_assets.items()},
                "asset_betas": betas,
                "average_correlation": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()),
                "correlation_range": {
                    "min": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min()),
                    "max": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max())
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error identifying risk factors: {e}")
            return {}
    
    def stress_test_portfolio(
        self, 
        returns_matrix: pd.DataFrame, 
        weights: Dict[str, float],
        scenarios: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Perform stress testing on portfolio under various scenarios.
        
        Args:
            returns_matrix: Historical returns data
            weights: Portfolio weights
            scenarios: Custom stress scenarios. If None, uses default scenarios.
            
        Returns:
            Dictionary with stress test results
        """
        
        if scenarios is None:
            scenarios = self._get_default_stress_scenarios()
        
        try:
            stress_results = {}
            
            # Align weights with returns
            aligned_returns = pd.DataFrame()
            weight_vector = []
            
            for asset, weight in weights.items():
                if asset in returns_matrix.columns:
                    aligned_returns[asset] = returns_matrix[asset]
                    weight_vector.append(weight)
            
            if aligned_returns.empty:
                return {}
            
            weight_vector = np.array(weight_vector)
            
            # Calculate baseline portfolio metrics
            baseline_returns = np.dot(aligned_returns.dropna().values, weight_vector)
            baseline_metrics = self.calculate_basic_risk_metrics(pd.Series(baseline_returns))
            
            # Apply each stress scenario
            for scenario_name, scenario_shocks in scenarios.items():
                scenario_returns = aligned_returns.copy()
                
                # Apply shocks to relevant assets
                for asset, shock in scenario_shocks.items():
                    if asset in scenario_returns.columns:
                        # Apply shock as a percentage change
                        scenario_returns[asset] = scenario_returns[asset] + (shock / 100)
                
                # Calculate stressed portfolio returns
                stressed_returns = np.dot(scenario_returns.dropna().values, weight_vector)
                stressed_metrics = self.calculate_basic_risk_metrics(pd.Series(stressed_returns))
                
                # Calculate impact
                impact = {
                    "return_impact": stressed_metrics["annualized_return"] - baseline_metrics["annualized_return"],
                    "volatility_impact": stressed_metrics["annualized_volatility"] - baseline_metrics["annualized_volatility"],
                    "sharpe_impact": stressed_metrics["sharpe_ratio"] - baseline_metrics["sharpe_ratio"],
                    "max_drawdown_impact": stressed_metrics["max_drawdown"] - baseline_metrics["max_drawdown"]
                }
                
                stress_results[scenario_name] = {
                    "scenario_shocks": scenario_shocks,
                    "stressed_metrics": stressed_metrics,
                    "impact": impact
                }
            
            return {
                "baseline_metrics": baseline_metrics,
                "stress_scenarios": stress_results,
                "worst_case_scenario": min(stress_results.keys(), 
                                         key=lambda x: stress_results[x]["stressed_metrics"]["annualized_return"]),
                "summary": {
                    "scenarios_tested": len(scenarios),
                    "worst_return_impact": min([s["impact"]["return_impact"] for s in stress_results.values()]),
                    "worst_drawdown_impact": min([s["impact"]["max_drawdown_impact"] for s in stress_results.values()])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in stress testing: {e}")
            return {}
    
    def _get_default_stress_scenarios(self) -> Dict[str, Dict[str, float]]:
        """Get default stress testing scenarios."""
        
        return {
            "market_crash": {
                "global_equity": -20,
                "us_equity": -25,
                "european_equity": -20,
                "emerging_markets": -30
            },
            "interest_rate_shock": {
                "real_estate": -15,
                "precious_metals": 5
            },
            "inflation_shock": {
                "precious_metals": 10,
                "real_estate": 5,
                "cryptocurrency": -20
            },
            "currency_crisis": {
                "global_equity": -10,
                "emerging_markets": -25,
                "precious_metals": 15
            },
            "tech_crash": {
                "us_equity": -30,
                "cryptocurrency": -50,
                "small_cap": -25
            }
        }
    
    def _empty_risk_metrics(self) -> Dict[str, Any]:
        """Return empty risk metrics structure."""
        return {
            "mean_return": 0.0,
            "std_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "downside_deviation": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "var_95": 0.0,
            "var_99": 0.0,
            "cvar_95": 0.0,
            "cvar_99": 0.0,
            "max_drawdown": 0.0,
            "positive_days": 0,
            "negative_days": 0,
            "total_days": 0,
            "win_rate": 0.0
        }
    
    def _empty_portfolio_risk_metrics(self) -> Dict[str, Any]:
        """Return empty portfolio risk metrics structure."""
        base_metrics = self._empty_risk_metrics()
        base_metrics.update({
            "portfolio_volatility": 0.0,
            "annualized_portfolio_volatility": 0.0,
            "diversification_ratio": 1.0,
            "number_of_assets": 0,
            "effective_number_of_assets": 0.0,
            "concentration_ratio": 0.0,
            "correlation_matrix": {},
            "weights": {}
        })
        return base_metrics