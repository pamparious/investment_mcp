"""Risk metrics calculator for portfolio and fund analysis."""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

from ...utils.exceptions import RiskCalculationError

logger = logging.getLogger(__name__)


class RiskMetricsCalculator:
    """Comprehensive risk metrics calculator for financial analysis."""
    
    def __init__(self, confidence_levels: Optional[List[float]] = None):
        """
        Initialize the risk metrics calculator.
        
        Args:
            confidence_levels: List of confidence levels for VaR calculations
        """
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def calculate_comprehensive_risk_metrics(
        self,
        fund_data: Dict[str, pd.DataFrame],
        portfolio_weights: Optional[Dict[str, float]] = None,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics for funds and portfolio.
        
        Args:
            fund_data: Dictionary of fund DataFrames
            portfolio_weights: Optional portfolio weights
            benchmark_data: Optional benchmark data for relative metrics
            
        Returns:
            Dictionary with comprehensive risk metrics
        """
        try:
            self.logger.info("Calculating comprehensive risk metrics")
            
            # Individual fund metrics
            fund_metrics = {}
            for fund_name, data in fund_data.items():
                if not data.empty and 'daily_return' in data.columns:
                    fund_metrics[fund_name] = self._calculate_fund_risk_metrics(
                        data['daily_return'].dropna(), fund_name
                    )
            
            # Portfolio metrics (if weights provided)
            portfolio_metrics = {}
            if portfolio_weights and fund_metrics:
                portfolio_metrics = self._calculate_portfolio_risk_metrics(
                    fund_data, portfolio_weights
                )
            
            # Correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(fund_data)
            
            # Risk decomposition
            risk_decomposition = self._calculate_risk_decomposition(
                fund_data, portfolio_weights
            )
            
            # Stress testing
            stress_test_results = self._perform_stress_tests(fund_data, portfolio_weights)
            
            # Market risk factors
            market_risk_factors = self._analyze_market_risk_factors(fund_data)
            
            # Tail risk analysis
            tail_risk_analysis = self._analyze_tail_risk(fund_data)
            
            results = {
                "individual_fund_metrics": fund_metrics,
                "portfolio_metrics": portfolio_metrics,
                "correlation_analysis": {
                    "correlation_matrix": correlation_matrix,
                    "diversification_ratio": self._calculate_diversification_ratio(correlation_matrix, portfolio_weights),
                    "concentration_risk": self._assess_concentration_risk(portfolio_weights)
                },
                "risk_decomposition": risk_decomposition,
                "stress_testing": stress_test_results,
                "market_risk_factors": market_risk_factors,
                "tail_risk_analysis": tail_risk_analysis,
                "risk_summary": self._generate_risk_summary(fund_metrics, portfolio_metrics)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            raise RiskCalculationError(f"Risk metrics calculation failed: {e}")
    
    def _calculate_fund_risk_metrics(
        self,
        returns: pd.Series,
        fund_name: str
    ) -> Dict[str, Any]:
        """Calculate risk metrics for a single fund."""
        
        if len(returns) < 30:
            return {"error": "Insufficient data for risk calculation"}
        
        returns_clean = returns.dropna()
        
        # Basic statistics
        metrics = {
            "volatility_annual": float(returns_clean.std() * np.sqrt(252)),
            "mean_return_annual": float(returns_clean.mean() * 252),
            "skewness": float(stats.skew(returns_clean)),
            "kurtosis": float(stats.kurtosis(returns_clean)),
            "jarque_bera_test": self._jarque_bera_test(returns_clean)
        }
        
        # Sharpe ratio
        risk_free_rate = 0.02  # 2% annual risk-free rate
        excess_returns = returns_clean - (risk_free_rate / 252)
        metrics["sharpe_ratio"] = float(
            (excess_returns.mean() * 252) / (returns_clean.std() * np.sqrt(252))
        )
        
        # Sortino ratio (downside deviation)
        downside_returns = returns_clean[returns_clean < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            metrics["sortino_ratio"] = float(
                (returns_clean.mean() * 252) / downside_deviation
            )
        else:
            metrics["sortino_ratio"] = np.inf
        
        # Maximum drawdown
        metrics["max_drawdown"] = self._calculate_max_drawdown(returns_clean)
        
        # Value at Risk (VaR)
        metrics["var"] = {}
        for confidence in self.confidence_levels:
            var_parametric = self._calculate_parametric_var(returns_clean, confidence)
            var_historical = self._calculate_historical_var(returns_clean, confidence)
            var_cornish_fisher = self._calculate_cornish_fisher_var(returns_clean, confidence)
            
            metrics["var"][f"{confidence:.0%}"] = {
                "parametric": float(var_parametric),
                "historical": float(var_historical),
                "cornish_fisher": float(var_cornish_fisher)
            }
        
        # Conditional Value at Risk (CVaR/Expected Shortfall)
        metrics["cvar"] = {}
        for confidence in self.confidence_levels:
            cvar = self._calculate_cvar(returns_clean, confidence)
            metrics["cvar"][f"{confidence:.0%}"] = float(cvar)
        
        # Rolling risk metrics
        metrics["rolling_metrics"] = self._calculate_rolling_risk_metrics(returns_clean)
        
        # Beta (if benchmark available)
        # Note: This would require benchmark data to be passed
        
        return metrics
    
    def _jarque_bera_test(self, returns: pd.Series) -> Dict[str, float]:
        """Perform Jarque-Bera normality test."""
        
        statistic, p_value = stats.jarque_bera(returns)
        
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_normal": bool(p_value > 0.05)
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate maximum drawdown metrics."""
        
        cumulative = (1 + returns.fillna(0)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_start = None
        max_dd_end = None
        max_dd_duration = 0
        
        # Find the drawdown period
        if not drawdown.empty:
            max_dd_idx = drawdown.idxmin()
            
            # Find start of drawdown (last peak before max drawdown)
            peak_before = running_max.loc[:max_dd_idx].idxmax()
            max_dd_start = peak_before
            
            # Find end of drawdown (recovery to peak)
            peak_value = running_max.loc[max_dd_idx]
            recovery_series = cumulative.loc[max_dd_idx:]
            recovery_idx = recovery_series[recovery_series >= peak_value].index
            
            if len(recovery_idx) > 0:
                max_dd_end = recovery_idx[0]
                max_dd_duration = (max_dd_end - max_dd_start).days
            else:
                max_dd_end = returns.index[-1]
                max_dd_duration = (max_dd_end - max_dd_start).days
        
        return {
            "max_drawdown": float(max_dd),
            "max_drawdown_start": max_dd_start.isoformat() if max_dd_start else None,
            "max_drawdown_end": max_dd_end.isoformat() if max_dd_end else None,
            "max_drawdown_duration_days": max_dd_duration,
            "current_drawdown": float(drawdown.iloc[-1]) if not drawdown.empty else 0.0
        }
    
    def _calculate_parametric_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate parametric VaR assuming normal distribution."""
        
        mean_return = returns.mean()
        std_return = returns.std()
        z_score = stats.norm.ppf(1 - confidence)
        
        return mean_return + z_score * std_return
    
    def _calculate_historical_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate historical VaR using empirical distribution."""
        
        return returns.quantile(1 - confidence)
    
    def _calculate_cornish_fisher_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Cornish-Fisher VaR accounting for skewness and kurtosis."""
        
        mean_return = returns.mean()
        std_return = returns.std()
        skewness = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        
        z = stats.norm.ppf(1 - confidence)
        
        # Cornish-Fisher expansion
        z_cf = (z + 
                (z**2 - 1) * skewness / 6 + 
                (z**3 - 3*z) * kurt / 24 - 
                (2*z**3 - 5*z) * (skewness**2) / 36)
        
        return mean_return + z_cf * std_return
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        
        var = self._calculate_historical_var(returns, confidence)
        tail_returns = returns[returns <= var]
        
        return tail_returns.mean() if len(tail_returns) > 0 else var
    
    def _calculate_rolling_risk_metrics(
        self,
        returns: pd.Series,
        windows: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Calculate rolling risk metrics."""
        
        if windows is None:
            windows = [30, 60, 252]
        
        rolling_metrics = {}
        
        for window in windows:
            if len(returns) >= window:
                rolling_vol = returns.rolling(window).std() * np.sqrt(252)
                rolling_var_95 = returns.rolling(window).quantile(0.05)
                
                rolling_metrics[f"{window}d"] = {
                    "volatility": {
                        "current": float(rolling_vol.iloc[-1]) if not rolling_vol.empty else None,
                        "mean": float(rolling_vol.mean()) if not rolling_vol.empty else None,
                        "std": float(rolling_vol.std()) if not rolling_vol.empty else None
                    },
                    "var_95": {
                        "current": float(rolling_var_95.iloc[-1]) if not rolling_var_95.empty else None,
                        "mean": float(rolling_var_95.mean()) if not rolling_var_95.empty else None
                    }
                }
        
        return rolling_metrics
    
    def _calculate_portfolio_risk_metrics(
        self,
        fund_data: Dict[str, pd.DataFrame],
        portfolio_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate portfolio-level risk metrics."""
        
        # Align all fund returns
        returns_data = {}
        for fund_name, data in fund_data.items():
            if fund_name in portfolio_weights and 'daily_return' in data.columns:
                returns_data[fund_name] = data['daily_return']
        
        if not returns_data:
            return {"error": "No valid return data for portfolio calculation"}
        
        # Create returns matrix
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if returns_df.empty:
            return {"error": "No overlapping return data"}
        
        # Calculate portfolio returns
        weights_vector = np.array([portfolio_weights.get(fund, 0) for fund in returns_df.columns])
        portfolio_returns = returns_df.dot(weights_vector)
        
        # Calculate portfolio risk metrics
        portfolio_metrics = self._calculate_fund_risk_metrics(portfolio_returns, "Portfolio")
        
        # Additional portfolio-specific metrics
        portfolio_metrics["diversification_metrics"] = self._calculate_diversification_metrics(
            returns_df, weights_vector
        )
        
        # Risk decomposition
        portfolio_metrics["risk_contribution"] = self._calculate_risk_contribution(
            returns_df, weights_vector
        )
        
        return portfolio_metrics
    
    def _calculate_diversification_metrics(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray
    ) -> Dict[str, float]:
        """Calculate portfolio diversification metrics."""
        
        # Portfolio volatility
        cov_matrix = returns_df.cov() * 252  # Annualized
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Weighted average volatility
        individual_vols = returns_df.std() * np.sqrt(252)
        weighted_avg_vol = np.dot(weights, individual_vols)
        
        # Diversification ratio
        diversification_ratio = weighted_avg_vol / portfolio_volatility
        
        # Effective number of assets
        effective_n_assets = 1 / np.sum(weights**2)
        
        return {
            "portfolio_volatility": float(portfolio_volatility),
            "weighted_average_volatility": float(weighted_avg_vol),
            "diversification_ratio": float(diversification_ratio),
            "effective_number_assets": float(effective_n_assets)
        }
    
    def _calculate_risk_contribution(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray
    ) -> Dict[str, float]:
        """Calculate individual asset risk contributions to portfolio."""
        
        cov_matrix = returns_df.cov() * 252
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        
        # Marginal risk contribution
        marginal_contrib = np.dot(cov_matrix, weights) / np.sqrt(portfolio_variance)
        
        # Component risk contribution
        component_contrib = weights * marginal_contrib
        
        # Percentage risk contribution
        pct_contrib = component_contrib / np.sum(component_contrib)
        
        risk_contributions = {}
        for i, fund_name in enumerate(returns_df.columns):
            risk_contributions[fund_name] = {
                "marginal_contribution": float(marginal_contrib[i]),
                "component_contribution": float(component_contrib[i]),
                "percentage_contribution": float(pct_contrib[i])
            }
        
        return risk_contributions
    
    def _calculate_correlation_matrix(
        self,
        fund_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Calculate and analyze correlation matrix."""
        
        # Collect returns data
        returns_data = {}
        for fund_name, data in fund_data.items():
            if 'daily_return' in data.columns:
                returns_data[fund_name] = data['daily_return']
        
        if len(returns_data) < 2:
            return {"error": "Need at least 2 funds for correlation analysis"}
        
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if returns_df.empty:
            return {"error": "No overlapping return data"}
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Analyze correlation structure
        correlation_analysis = {
            "correlation_matrix": corr_matrix.to_dict(),
            "average_correlation": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()),
            "max_correlation": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()),
            "min_correlation": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min()),
            "correlation_distribution": self._analyze_correlation_distribution(corr_matrix)
        }
        
        # Principal component analysis
        try:
            pca = PCA()
            pca.fit(returns_df.fillna(0))
            
            correlation_analysis["pca_analysis"] = {
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_).tolist(),
                "first_pc_variance": float(pca.explained_variance_ratio_[0]),
                "components_for_95pct": int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95)) + 1
            }
        except Exception as e:
            self.logger.warning(f"PCA analysis failed: {e}")
            correlation_analysis["pca_analysis"] = {"error": str(e)}
        
        return correlation_analysis
    
    def _analyze_correlation_distribution(self, corr_matrix: pd.DataFrame) -> Dict[str, float]:
        """Analyze the distribution of correlations."""
        
        # Get upper triangle correlations (excluding diagonal)
        correlations = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        
        return {
            "mean": float(np.mean(correlations)),
            "median": float(np.median(correlations)),
            "std": float(np.std(correlations)),
            "q25": float(np.percentile(correlations, 25)),
            "q75": float(np.percentile(correlations, 75)),
            "high_correlation_count": int(np.sum(correlations > 0.7)),
            "negative_correlation_count": int(np.sum(correlations < 0))
        }
    
    def _calculate_diversification_ratio(
        self,
        correlation_matrix: Dict[str, Any],
        portfolio_weights: Optional[Dict[str, float]]
    ) -> Optional[float]:
        """Calculate portfolio diversification ratio."""
        
        if not portfolio_weights or "correlation_matrix" not in correlation_matrix:
            return None
        
        try:
            corr_df = pd.DataFrame(correlation_matrix["correlation_matrix"])
            
            # Filter weights to match correlation matrix
            filtered_weights = {k: v for k, v in portfolio_weights.items() if k in corr_df.columns}
            
            if not filtered_weights:
                return None
            
            weights_array = np.array([filtered_weights.get(col, 0) for col in corr_df.columns])
            
            # Assume equal volatility for simplification
            weighted_avg_vol = np.sum(weights_array)  # If all vols = 1
            portfolio_vol = np.sqrt(np.dot(weights_array, np.dot(corr_df.values, weights_array)))
            
            return float(weighted_avg_vol / portfolio_vol)
            
        except Exception as e:
            self.logger.warning(f"Diversification ratio calculation failed: {e}")
            return None
    
    def _assess_concentration_risk(self, portfolio_weights: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """Assess portfolio concentration risk."""
        
        if not portfolio_weights:
            return {"error": "No portfolio weights provided"}
        
        weights = np.array(list(portfolio_weights.values()))
        weights = weights / np.sum(weights)  # Normalize
        
        # Herfindahl-Hirschman Index
        hhi = np.sum(weights**2)
        
        # Effective number of assets
        effective_n = 1 / hhi
        
        # Concentration categories
        if hhi > 0.25:
            concentration_level = "high"
        elif hhi > 0.15:
            concentration_level = "medium"
        else:
            concentration_level = "low"
        
        # Top N concentration
        sorted_weights = np.sort(weights)[::-1]
        top_3_concentration = np.sum(sorted_weights[:3])
        top_5_concentration = np.sum(sorted_weights[:5])
        
        return {
            "herfindahl_index": float(hhi),
            "effective_number_assets": float(effective_n),
            "concentration_level": concentration_level,
            "top_3_concentration": float(top_3_concentration),
            "top_5_concentration": float(top_5_concentration),
            "max_weight": float(np.max(weights)),
            "min_weight": float(np.min(weights))
        }
    
    def _calculate_risk_decomposition(
        self,
        fund_data: Dict[str, pd.DataFrame],
        portfolio_weights: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Decompose portfolio risk into systematic and idiosyncratic components."""
        
        if not portfolio_weights:
            return {"error": "Portfolio weights required for risk decomposition"}
        
        try:
            # Collect returns data
            returns_data = {}
            for fund_name, data in fund_data.items():
                if fund_name in portfolio_weights and 'daily_return' in data.columns:
                    returns_data[fund_name] = data['daily_return']
            
            if len(returns_data) < 2:
                return {"error": "Need at least 2 funds for risk decomposition"}
            
            returns_df = pd.DataFrame(returns_data).dropna()
            
            if returns_df.empty:
                return {"error": "No overlapping return data"}
            
            # Use Ledoit-Wolf shrinkage estimator for covariance matrix
            lw = LedoitWolf()
            cov_matrix = lw.fit(returns_df).covariance_ * 252  # Annualized
            
            # Portfolio weights
            weights = np.array([portfolio_weights.get(fund, 0) for fund in returns_df.columns])
            weights = weights / np.sum(weights)  # Normalize
            
            # Portfolio variance
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            
            # Factor model approach using PCA
            pca = PCA(n_components=min(5, len(returns_df.columns)))
            factor_loadings = pca.fit_transform(returns_df.fillna(0).T)
            explained_variance = pca.explained_variance_
            
            # Systematic risk (first few principal components)
            systematic_components = min(3, len(explained_variance))
            systematic_variance = np.sum(explained_variance[:systematic_components])
            total_variance = np.sum(explained_variance)
            
            systematic_risk_ratio = systematic_variance / total_variance if total_variance > 0 else 0
            
            decomposition = {
                "total_portfolio_risk": float(np.sqrt(portfolio_variance)),
                "systematic_risk_ratio": float(systematic_risk_ratio),
                "idiosyncratic_risk_ratio": float(1 - systematic_risk_ratio),
                "factor_contributions": {
                    f"factor_{i+1}": float(explained_variance[i] / total_variance)
                    for i in range(min(5, len(explained_variance)))
                },
                "shrinkage_intensity": float(lw.shrinkage_)
            }
            
            return decomposition
            
        except Exception as e:
            self.logger.error(f"Risk decomposition failed: {e}")
            return {"error": str(e)}
    
    def _perform_stress_tests(
        self,
        fund_data: Dict[str, pd.DataFrame],
        portfolio_weights: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Perform various stress tests on the portfolio."""
        
        stress_results = {}
        
        # Historical stress scenarios
        stress_results["historical_scenarios"] = self._historical_stress_scenarios(fund_data, portfolio_weights)
        
        # Monte Carlo stress testing
        stress_results["monte_carlo"] = self._monte_carlo_stress_test(fund_data, portfolio_weights)
        
        # Correlation breakdown scenarios
        stress_results["correlation_stress"] = self._correlation_stress_test(fund_data, portfolio_weights)
        
        return stress_results
    
    def _historical_stress_scenarios(
        self,
        fund_data: Dict[str, pd.DataFrame],
        portfolio_weights: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Test portfolio against historical stress periods."""
        
        if not portfolio_weights:
            return {"error": "Portfolio weights required"}
        
        # Define historical stress periods (example dates)
        stress_periods = {
            "covid_crash_2020": ("2020-02-20", "2020-03-20"),
            "eurozone_crisis_2011": ("2011-07-01", "2011-09-30"),
            "financial_crisis_2008": ("2008-09-01", "2008-11-30")
        }
        
        scenario_results = {}
        
        for scenario_name, (start_date, end_date) in stress_periods.items():
            try:
                scenario_returns = []
                
                for fund_name, data in fund_data.items():
                    if fund_name in portfolio_weights and 'daily_return' in data.columns:
                        period_data = data.loc[start_date:end_date, 'daily_return']
                        if not period_data.empty:
                            scenario_returns.append(period_data * portfolio_weights[fund_name])
                
                if scenario_returns:
                    portfolio_returns = pd.concat(scenario_returns, axis=1).sum(axis=1)
                    
                    scenario_results[scenario_name] = {
                        "total_return": float(portfolio_returns.sum()),
                        "worst_day": float(portfolio_returns.min()),
                        "volatility": float(portfolio_returns.std() * np.sqrt(252)),
                        "max_drawdown": float(self._calculate_max_drawdown(portfolio_returns)["max_drawdown"])
                    }
                else:
                    scenario_results[scenario_name] = {"error": "No data for period"}
                    
            except Exception as e:
                scenario_results[scenario_name] = {"error": str(e)}
        
        return scenario_results
    
    def _monte_carlo_stress_test(
        self,
        fund_data: Dict[str, pd.DataFrame],
        portfolio_weights: Optional[Dict[str, float]],
        n_simulations: int = 10000
    ) -> Dict[str, Any]:
        """Perform Monte Carlo stress testing."""
        
        if not portfolio_weights:
            return {"error": "Portfolio weights required"}
        
        try:
            # Collect returns data
            returns_data = {}
            for fund_name, data in fund_data.items():
                if fund_name in portfolio_weights and 'daily_return' in data.columns:
                    returns_data[fund_name] = data['daily_return']
            
            if not returns_data:
                return {"error": "No valid return data"}
            
            returns_df = pd.DataFrame(returns_data).dropna()
            
            if returns_df.empty:
                return {"error": "No overlapping return data"}
            
            # Calculate mean and covariance
            mean_returns = returns_df.mean().values
            cov_matrix = returns_df.cov().values
            
            # Portfolio weights
            weights = np.array([portfolio_weights.get(fund, 0) for fund in returns_df.columns])
            weights = weights / np.sum(weights)
            
            # Monte Carlo simulation
            np.random.seed(42)  # For reproducibility
            simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_simulations)
            portfolio_returns = simulated_returns @ weights
            
            # Calculate stress metrics
            var_95 = np.percentile(portfolio_returns, 5)
            var_99 = np.percentile(portfolio_returns, 1)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()
            
            monte_carlo_results = {
                "simulations": n_simulations,
                "var_95": float(var_95),
                "var_99": float(var_99),
                "cvar_95": float(cvar_95),
                "cvar_99": float(cvar_99),
                "worst_case_1pct": float(np.percentile(portfolio_returns, 1)),
                "best_case_1pct": float(np.percentile(portfolio_returns, 99)),
                "probability_negative": float(np.mean(portfolio_returns < 0))
            }
            
            return monte_carlo_results
            
        except Exception as e:
            self.logger.error(f"Monte Carlo stress test failed: {e}")
            return {"error": str(e)}
    
    def _correlation_stress_test(
        self,
        fund_data: Dict[str, pd.DataFrame],
        portfolio_weights: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Test portfolio under correlation breakdown scenarios."""
        
        if not portfolio_weights:
            return {"error": "Portfolio weights required"}
        
        try:
            # Collect returns data
            returns_data = {}
            for fund_name, data in fund_data.items():
                if fund_name in portfolio_weights and 'daily_return' in data.columns:
                    returns_data[fund_name] = data['daily_return']
            
            returns_df = pd.DataFrame(returns_data).dropna()
            
            if returns_df.empty or len(returns_df.columns) < 2:
                return {"error": "Insufficient data for correlation stress test"}
            
            # Original correlation matrix
            original_corr = returns_df.corr()
            
            # Stress scenarios
            stress_scenarios = {
                "all_correlations_to_1": np.ones_like(original_corr.values),
                "all_correlations_to_0": np.eye(len(original_corr)),
                "correlations_increase_50pct": original_corr.values * 1.5
            }
            
            # Ensure diagonal is 1 and values are valid correlations
            for scenario_name, corr_matrix in stress_scenarios.items():
                np.fill_diagonal(corr_matrix, 1.0)
                corr_matrix = np.clip(corr_matrix, -1, 1)
                stress_scenarios[scenario_name] = corr_matrix
            
            weights = np.array([portfolio_weights.get(fund, 0) for fund in returns_df.columns])
            weights = weights / np.sum(weights)
            
            # Calculate portfolio volatility under each scenario
            individual_vols = returns_df.std().values
            cov_base = np.outer(individual_vols, individual_vols)
            
            scenario_results = {}
            original_vol = np.sqrt(np.dot(weights, np.dot(original_corr.values * cov_base, weights)))
            
            for scenario_name, stress_corr in stress_scenarios.items():
                stress_cov = stress_corr * cov_base
                stress_vol = np.sqrt(np.dot(weights, np.dot(stress_cov, weights)))
                
                scenario_results[scenario_name] = {
                    "portfolio_volatility": float(stress_vol * np.sqrt(252)),
                    "volatility_change": float((stress_vol - original_vol) / original_vol),
                    "risk_increase_pct": float(((stress_vol / original_vol) - 1) * 100)
                }
            
            scenario_results["original_volatility"] = float(original_vol * np.sqrt(252))
            
            return scenario_results
            
        except Exception as e:
            self.logger.error(f"Correlation stress test failed: {e}")
            return {"error": str(e)}
    
    def _analyze_market_risk_factors(self, fund_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze exposure to common market risk factors."""
        
        try:
            # Collect all returns
            returns_data = {}
            for fund_name, data in fund_data.items():
                if 'daily_return' in data.columns:
                    returns_data[fund_name] = data['daily_return']
            
            if len(returns_data) < 2:
                return {"error": "Need at least 2 funds for factor analysis"}
            
            returns_df = pd.DataFrame(returns_data).dropna()
            
            if returns_df.empty:
                return {"error": "No overlapping return data"}
            
            # Principal Component Analysis for factor identification
            pca = PCA()
            pca.fit(returns_df.fillna(0))
            
            # Factor loadings
            factor_loadings = pca.components_[:5]  # Top 5 factors
            
            factor_analysis = {
                "explained_variance_ratio": pca.explained_variance_ratio_[:5].tolist(),
                "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_[:5]).tolist(),
                "factor_loadings": {}
            }
            
            # Interpret factors based on loadings
            for i, loadings in enumerate(factor_loadings):
                factor_analysis["factor_loadings"][f"factor_{i+1}"] = {
                    fund_name: float(loading)
                    for fund_name, loading in zip(returns_df.columns, loadings)
                }
            
            # Market beta estimation (using first principal component as market proxy)
            market_factor = pca.transform(returns_df.fillna(0))[:, 0]
            
            betas = {}
            for fund_name in returns_df.columns:
                fund_returns = returns_df[fund_name].dropna()
                
                # Align data
                aligned_data = pd.concat([fund_returns, pd.Series(market_factor, index=returns_df.index)], axis=1).dropna()
                
                if len(aligned_data) > 30:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        aligned_data.iloc[:, 1], aligned_data.iloc[:, 0]
                    )
                    
                    betas[fund_name] = {
                        "beta": float(slope),
                        "alpha": float(intercept),
                        "r_squared": float(r_value**2),
                        "p_value": float(p_value)
                    }
            
            factor_analysis["market_betas"] = betas
            
            return factor_analysis
            
        except Exception as e:
            self.logger.error(f"Market risk factor analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_tail_risk(self, fund_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze tail risk characteristics."""
        
        tail_analysis = {}
        
        for fund_name, data in fund_data.items():
            if 'daily_return' in data.columns:
                returns = data['daily_return'].dropna()
                
                if len(returns) > 100:
                    tail_metrics = self._calculate_tail_metrics(returns)
                    tail_analysis[fund_name] = tail_metrics
        
        # Aggregate tail risk metrics
        if tail_analysis:
            aggregate_tail = self._aggregate_tail_metrics(tail_analysis)
            tail_analysis["aggregate_metrics"] = aggregate_tail
        
        return tail_analysis
    
    def _calculate_tail_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate tail risk metrics for a return series."""
        
        # Tail ratios
        left_tail = returns.quantile(0.05)
        right_tail = returns.quantile(0.95)
        
        tail_metrics = {
            "left_tail_5pct": float(left_tail),
            "right_tail_5pct": float(right_tail),
            "tail_ratio": float(abs(left_tail) / right_tail) if right_tail != 0 else None
        }
        
        # Extreme value analysis
        try:
            from scipy.stats import genextreme
            
            # Fit generalized extreme value distribution to negative returns (losses)
            losses = -returns[returns < 0]
            if len(losses) > 20:
                params = genextreme.fit(losses)
                tail_metrics["gev_parameters"] = {
                    "shape": float(params[0]),
                    "location": float(params[1]),
                    "scale": float(params[2])
                }
                
                # Estimate extreme quantiles
                tail_metrics["extreme_quantiles"] = {
                    "loss_99_9pct": float(-genextreme.ppf(0.999, *params)),
                    "loss_99_99pct": float(-genextreme.ppf(0.9999, *params))
                }
        except Exception as e:
            self.logger.debug(f"Extreme value analysis failed: {e}")
            tail_metrics["extreme_value_analysis"] = {"error": str(e)}
        
        return tail_metrics
    
    def _aggregate_tail_metrics(self, individual_tail_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate tail metrics across all funds."""
        
        left_tails = []
        tail_ratios = []
        
        for fund_metrics in individual_tail_metrics.values():
            if isinstance(fund_metrics, dict):
                if "left_tail_5pct" in fund_metrics:
                    left_tails.append(fund_metrics["left_tail_5pct"])
                if "tail_ratio" in fund_metrics and fund_metrics["tail_ratio"] is not None:
                    tail_ratios.append(fund_metrics["tail_ratio"])
        
        aggregate = {}
        
        if left_tails:
            aggregate["average_left_tail"] = float(np.mean(left_tails))
            aggregate["worst_left_tail"] = float(np.min(left_tails))
            
        if tail_ratios:
            aggregate["average_tail_ratio"] = float(np.mean(tail_ratios))
            aggregate["max_tail_ratio"] = float(np.max(tail_ratios))
        
        return aggregate
    
    def _generate_risk_summary(
        self,
        fund_metrics: Dict[str, Any],
        portfolio_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate executive summary of risk analysis."""
        
        summary = {
            "funds_analyzed": len(fund_metrics),
            "risk_assessment": "unknown"
        }
        
        if fund_metrics:
            # Average metrics across funds
            volatilities = []
            sharpe_ratios = []
            max_drawdowns = []
            
            for metrics in fund_metrics.values():
                if isinstance(metrics, dict) and "error" not in metrics:
                    if "volatility_annual" in metrics:
                        volatilities.append(metrics["volatility_annual"])
                    if "sharpe_ratio" in metrics:
                        sharpe_ratios.append(metrics["sharpe_ratio"])
                    if "max_drawdown" in metrics and isinstance(metrics["max_drawdown"], dict):
                        max_drawdowns.append(abs(metrics["max_drawdown"].get("max_drawdown", 0)))
            
            if volatilities:
                avg_vol = np.mean(volatilities)
                summary["average_volatility"] = float(avg_vol)
                
                if avg_vol > 0.25:
                    summary["risk_assessment"] = "high"
                elif avg_vol > 0.15:
                    summary["risk_assessment"] = "medium"
                else:
                    summary["risk_assessment"] = "low"
            
            if sharpe_ratios:
                summary["average_sharpe_ratio"] = float(np.mean(sharpe_ratios))
            
            if max_drawdowns:
                summary["average_max_drawdown"] = float(np.mean(max_drawdowns))
        
        if portfolio_metrics and "error" not in portfolio_metrics:
            summary["portfolio_risk_metrics"] = {
                "volatility": portfolio_metrics.get("volatility_annual"),
                "sharpe_ratio": portfolio_metrics.get("sharpe_ratio"),
                "max_drawdown": portfolio_metrics.get("max_drawdown", {}).get("max_drawdown")
            }
        
        return summary