"""Risk analysis and metrics calculations."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RiskAnalyzer:
    """Risk analysis calculations and metrics."""
    
    def __init__(self):
        """Initialize the risk analyzer."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def prepare_returns_data(self, market_data: List[Dict[str, Any]]) -> pd.Series:
        """
        Prepare returns data from market data.
        
        Args:
            market_data: List of market data dictionaries
            
        Returns:
            Pandas Series of daily returns
        """
        if not market_data:
            return pd.Series()
        
        try:
            df = pd.DataFrame(market_data)
            
            if 'date' not in df.columns or 'close_price' not in df.columns:
                self.logger.warning("Missing required columns for returns calculation")
                return pd.Series()
            
            # Convert date and sort
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
            
            # Calculate daily returns
            returns = df['close_price'].pct_change().dropna()
            returns.index = df['date'][1:]  # Align with returns
            
            return returns
            
        except Exception as e:
            self.logger.error(f"Error preparing returns data: {e}")
            return pd.Series()
    
    def calculate_volatility(
        self, 
        market_data: List[Dict[str, Any]], 
        period: int = 252
    ) -> Dict[str, Any]:
        """
        Calculate price volatility metrics.
        
        Args:
            market_data: Historical price data
            period: Annualization period (252 for daily data)
            
        Returns:
            Dictionary containing volatility metrics
        """
        try:
            returns = self.prepare_returns_data(market_data)
            
            if returns.empty:
                return {"error": "Insufficient data for volatility calculation"}
            
            # Historical volatility (annualized standard deviation)
            historical_vol = returns.std() * np.sqrt(period)
            
            # Calculate rolling volatilities for trend analysis
            if len(returns) >= 30:
                rolling_30d = returns.rolling(window=30).std() * np.sqrt(period)
                current_30d_vol = rolling_30d.iloc[-1] if not pd.isna(rolling_30d.iloc[-1]) else historical_vol
            else:
                current_30d_vol = historical_vol
            
            if len(returns) >= 90:
                rolling_90d = returns.rolling(window=90).std() * np.sqrt(period)
                current_90d_vol = rolling_90d.iloc[-1] if not pd.isna(rolling_90d.iloc[-1]) else historical_vol
            else:
                current_90d_vol = historical_vol
            
            # Volatility classification
            if historical_vol < 0.15:
                vol_class = "low"
                risk_level = "conservative"
            elif historical_vol < 0.25:
                vol_class = "moderate"
                risk_level = "moderate"
            elif historical_vol < 0.40:
                vol_class = "high"
                risk_level = "aggressive"
            else:
                vol_class = "very high"
                risk_level = "speculative"
            
            # Volatility trend
            if current_30d_vol > current_90d_vol * 1.2:
                vol_trend = "increasing"
            elif current_30d_vol < current_90d_vol * 0.8:
                vol_trend = "decreasing"
            else:
                vol_trend = "stable"
            
            return {
                "historical_volatility": float(historical_vol),
                "current_30d_volatility": float(current_30d_vol),
                "current_90d_volatility": float(current_90d_vol),
                "volatility_class": vol_class,
                "risk_level": risk_level,
                "volatility_trend": vol_trend,
                "data_points": len(returns),
                "annualization_period": period
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return {"error": str(e)}
    
    def value_at_risk(
        self, 
        market_data: List[Dict[str, Any]], 
        confidence_level: float = 0.05,
        time_horizon: int = 1
    ) -> Dict[str, Any]:
        """
        Calculate Value at Risk (VaR) metrics.
        
        Args:
            market_data: Historical price data
            confidence_level: Confidence level for VaR (0.05 = 5%)
            time_horizon: Time horizon in days
            
        Returns:
            Dictionary containing VaR calculations
        """
        try:
            returns = self.prepare_returns_data(market_data)
            
            if returns.empty or len(returns) < 30:
                return {"error": "Insufficient data for VaR calculation (minimum 30 observations)"}
            
            # Get current price for absolute VaR calculation
            current_price = None
            if market_data:
                sorted_data = sorted(market_data, key=lambda x: pd.to_datetime(x['date']))
                current_price = float(sorted_data[-1].get('close_price', 0))
            
            # Historical VaR (percentile method)
            var_percentile = np.percentile(returns, confidence_level * 100)
            
            # Parametric VaR (assuming normal distribution)
            mean_return = returns.mean()
            std_return = returns.std()
            var_parametric = mean_return - (1.96 * std_return)  # 95% confidence
            
            # Adjust for time horizon
            var_percentile_adjusted = var_percentile * np.sqrt(time_horizon)
            var_parametric_adjusted = var_parametric * np.sqrt(time_horizon)
            
            # Convert to absolute dollar amounts if current price available
            absolute_var = {}
            if current_price and current_price > 0:
                absolute_var = {
                    "historical_var_dollar": float(current_price * abs(var_percentile_adjusted)),
                    "parametric_var_dollar": float(current_price * abs(var_parametric_adjusted)),
                    "current_price": float(current_price)
                }
            
            # Expected Shortfall (Conditional VaR)
            tail_returns = returns[returns <= var_percentile]
            expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else var_percentile
            
            # Risk assessment
            if abs(var_percentile) < 0.02:
                risk_assessment = "low"
            elif abs(var_percentile) < 0.04:
                risk_assessment = "moderate"
            elif abs(var_percentile) < 0.06:
                risk_assessment = "high"
            else:
                risk_assessment = "very high"
            
            result = {
                "confidence_level": confidence_level,
                "time_horizon_days": time_horizon,
                "historical_var": float(var_percentile_adjusted),
                "parametric_var": float(var_parametric_adjusted),
                "expected_shortfall": float(expected_shortfall * np.sqrt(time_horizon)),
                "risk_assessment": risk_assessment,
                "data_points": len(returns),
                "calculation_period": {
                    "start": returns.index.min().isoformat(),
                    "end": returns.index.max().isoformat()
                }
            }
            
            # Add absolute dollar amounts if available
            result.update(absolute_var)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            return {"error": str(e)}
    
    def maximum_drawdown(self, market_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate maximum drawdown - worst-case loss scenarios.
        
        Args:
            market_data: Historical price data
            
        Returns:
            Dictionary containing drawdown analysis
        """
        try:
            if not market_data:
                return {"error": "No market data provided"}
            
            df = pd.DataFrame(market_data)
            
            if 'date' not in df.columns or 'close_price' not in df.columns:
                return {"error": "Missing required columns"}
            
            # Prepare data
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
            df = df.dropna()
            
            if len(df) < 2:
                return {"error": "Insufficient data for drawdown calculation"}
            
            # Calculate cumulative returns
            prices = df['close_price']
            
            # Calculate running maximum (peak)
            running_max = prices.expanding().max()
            
            # Calculate drawdown from peak
            drawdown = (prices - running_max) / running_max
            
            # Find maximum drawdown
            max_dd = drawdown.min()
            max_dd_idx = drawdown.idxmin()
            
            # Find the peak before max drawdown
            peak_idx = running_max[:max_dd_idx].idxmax()
            
            # Find recovery point (if any)
            recovery_idx = None
            if max_dd_idx < len(df) - 1:
                peak_price = df.loc[peak_idx, 'close_price']
                post_dd_data = df.loc[max_dd_idx:]
                recovery_data = post_dd_data[post_dd_data['close_price'] >= peak_price]
                if not recovery_data.empty:
                    recovery_idx = recovery_data.index[0]
            
            # Calculate drawdown duration
            peak_date = df.loc[peak_idx, 'date']
            trough_date = df.loc[max_dd_idx, 'date']
            dd_duration = (trough_date - peak_date).days
            
            # Recovery duration
            recovery_duration = None
            if recovery_idx is not None:
                recovery_date = df.loc[recovery_idx, 'date']
                recovery_duration = (recovery_date - trough_date).days
            
            # Calculate current drawdown
            current_price = prices.iloc[-1]
            current_peak = running_max.iloc[-1]
            current_dd = (current_price - current_peak) / current_peak
            
            # Drawdown statistics
            drawdown_periods = []
            in_drawdown = False
            dd_start = None
            
            for i, dd_val in enumerate(drawdown):
                if dd_val < -0.05 and not in_drawdown:  # Start of significant drawdown (>5%)
                    in_drawdown = True
                    dd_start = i
                elif dd_val >= -0.01 and in_drawdown:  # End of drawdown (recovery to within 1%)
                    in_drawdown = False
                    if dd_start is not None:
                        period_dd = drawdown[dd_start:i+1].min()
                        period_duration = i - dd_start
                        drawdown_periods.append({
                            "max_drawdown": float(period_dd),
                            "duration_days": period_duration
                        })
            
            # Risk assessment
            if abs(max_dd) < 0.10:
                risk_level = "low"
            elif abs(max_dd) < 0.20:
                risk_level = "moderate"
            elif abs(max_dd) < 0.35:
                risk_level = "high"
            else:
                risk_level = "very high"
            
            result = {
                "maximum_drawdown": float(max_dd),
                "max_dd_percentage": float(max_dd * 100),
                "current_drawdown": float(current_dd),
                "current_dd_percentage": float(current_dd * 100),
                "drawdown_duration_days": dd_duration,
                "recovery_duration_days": recovery_duration,
                "risk_level": risk_level,
                "peak_date": peak_date.isoformat(),
                "trough_date": trough_date.isoformat(),
                "peak_price": float(df.loc[peak_idx, 'close_price']),
                "trough_price": float(df.loc[max_dd_idx, 'close_price']),
                "current_price": float(current_price),
                "historical_drawdowns": drawdown_periods,
                "analysis_period": {
                    "start": df['date'].min().isoformat(),
                    "end": df['date'].max().isoformat()
                }
            }
            
            if recovery_idx is not None:
                result["recovery_date"] = df.loc[recovery_idx, 'date'].isoformat()
                result["recovery_price"] = float(df.loc[recovery_idx, 'close_price'])
            else:
                result["recovery_status"] = "not_recovered"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating maximum drawdown: {e}")
            return {"error": str(e)}
    
    def portfolio_risk_metrics(
        self, 
        portfolio_data: Dict[str, List[Dict[str, Any]]], 
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate portfolio-level risk metrics.
        
        Args:
            portfolio_data: Dictionary of asset data by symbol
            weights: Optional portfolio weights (equal weight if not provided)
            
        Returns:
            Portfolio risk analysis
        """
        try:
            if not portfolio_data:
                return {"error": "No portfolio data provided"}
            
            # Prepare returns data for each asset
            returns_data = {}
            for symbol, data in portfolio_data.items():
                returns = self.prepare_returns_data(data)
                if not returns.empty:
                    returns_data[symbol] = returns
            
            if len(returns_data) < 2:
                return {"error": "Need at least 2 assets for portfolio risk analysis"}
            
            # Create combined returns DataFrame
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 30:
                return {"error": "Insufficient overlapping data for portfolio analysis"}
            
            # Set equal weights if not provided
            if weights is None:
                n_assets = len(returns_data)
                weights = {symbol: 1.0 / n_assets for symbol in returns_data.keys()}
            
            # Ensure weights sum to 1
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            # Calculate portfolio returns
            weight_series = pd.Series([weights.get(col, 0) for col in returns_df.columns], 
                                    index=returns_df.columns)
            portfolio_returns = (returns_df * weight_series).sum(axis=1)
            
            # Portfolio volatility
            portfolio_vol = portfolio_returns.std() * np.sqrt(252)
            
            # Individual asset volatilities
            individual_vols = returns_df.std() * np.sqrt(252)
            weighted_avg_vol = (individual_vols * weight_series).sum()
            
            # Diversification benefit
            diversification_ratio = portfolio_vol / weighted_avg_vol
            diversification_benefit = 1 - diversification_ratio
            
            # Portfolio VaR
            portfolio_var = np.percentile(portfolio_returns, 5)  # 5% VaR
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            
            # Portfolio beta (if benchmark data available)
            # For now, calculate relative to equal-weighted portfolio
            market_returns = returns_df.mean(axis=1)
            portfolio_beta = portfolio_returns.cov(market_returns) / market_returns.var()
            
            # Risk-adjusted returns
            portfolio_mean_return = portfolio_returns.mean() * 252  # Annualized
            sharpe_ratio = portfolio_mean_return / portfolio_vol if portfolio_vol > 0 else 0
            
            # Risk contribution by asset
            risk_contributions = {}
            for symbol in returns_df.columns:
                asset_weight = weights.get(symbol, 0)
                asset_vol = individual_vols[symbol]
                # Simplified risk contribution
                risk_contributions[symbol] = float(asset_weight * asset_vol / portfolio_vol) if portfolio_vol > 0 else 0
            
            return {
                "portfolio_volatility": float(portfolio_vol),
                "weighted_average_volatility": float(weighted_avg_vol),
                "diversification_ratio": float(diversification_ratio),
                "diversification_benefit": float(diversification_benefit),
                "portfolio_var_5pct": float(portfolio_var),
                "portfolio_beta": float(portfolio_beta),
                "annualized_return": float(portfolio_mean_return),
                "sharpe_ratio": float(sharpe_ratio),
                "average_correlation": float(avg_correlation),
                "risk_contributions": risk_contributions,
                "portfolio_weights": weights,
                "assets_analyzed": list(returns_data.keys()),
                "data_points": len(returns_df),
                "analysis_period": {
                    "start": returns_df.index.min().isoformat(),
                    "end": returns_df.index.max().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk metrics: {e}")
            return {"error": str(e)}
    
    def stress_test_analysis(
        self, 
        market_data: List[Dict[str, Any]], 
        stress_scenarios: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Perform stress testing analysis.
        
        Args:
            market_data: Historical price data
            stress_scenarios: List of stress scenario returns (default: -10%, -20%, -30%)
            
        Returns:
            Stress test results
        """
        try:
            if stress_scenarios is None:
                stress_scenarios = [-0.10, -0.20, -0.30]  # 10%, 20%, 30% declines
            
            returns = self.prepare_returns_data(market_data)
            
            if returns.empty:
                return {"error": "Insufficient data for stress testing"}
            
            # Get current price
            current_price = None
            if market_data:
                sorted_data = sorted(market_data, key=lambda x: pd.to_datetime(x['date']))
                current_price = float(sorted_data[-1].get('close_price', 0))
            
            # Historical stress events (worst single-day returns)
            worst_days = returns.nsmallest(5)
            
            # Scenario analysis
            scenario_results = []
            for scenario in stress_scenarios:
                scenario_price = current_price * (1 + scenario) if current_price else None
                
                # How often has this scenario occurred historically?
                worse_days = len(returns[returns <= scenario])
                probability = worse_days / len(returns) if len(returns) > 0 else 0
                
                scenario_results.append({
                    "scenario_return": float(scenario),
                    "scenario_percentage": float(scenario * 100),
                    "scenario_price": float(scenario_price) if scenario_price else None,
                    "historical_occurrences": int(worse_days),
                    "historical_probability": float(probability),
                    "annualized_probability": float(probability * 252)  # Approximate annual probability
                })
            
            # Tail risk analysis
            returns_5pct = returns[returns <= np.percentile(returns, 5)]
            tail_risk_avg = returns_5pct.mean() if len(returns_5pct) > 0 else 0
            
            return {
                "stress_scenarios": scenario_results,
                "worst_historical_days": [
                    {
                        "return": float(ret),
                        "percentage": float(ret * 100),
                        "date": date.isoformat()
                    }
                    for ret, date in zip(worst_days.values, worst_days.index)
                ],
                "tail_risk_5pct": float(tail_risk_avg),
                "current_price": float(current_price) if current_price else None,
                "analysis_period": {
                    "start": returns.index.min().isoformat(),
                    "end": returns.index.max().isoformat()
                },
                "data_points": len(returns)
            }
            
        except Exception as e:
            self.logger.error(f"Error in stress test analysis: {e}")
            return {"error": str(e)}