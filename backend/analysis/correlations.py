"""Correlation analysis for investment data."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Correlation analysis between different financial instruments and indicators."""
    
    def __init__(self):
        """Initialize the correlation analyzer."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def prepare_correlation_data(
        self, 
        market_data: Dict[str, List[Dict[str, Any]]], 
        economic_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for correlation analysis.
        
        Args:
            market_data: Dictionary of market data by symbol
            economic_data: Dictionary of economic data by type
            
        Returns:
            Dictionary of prepared DataFrames
        """
        prepared_data = {}
        
        try:
            # Prepare market data
            for symbol, data in market_data.items():
                if data:
                    df = pd.DataFrame(data)
                    if 'date' in df.columns and 'close_price' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date').sort_index()
                        df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
                        prepared_data[f"market_{symbol}"] = df[['close_price']].rename(columns={'close_price': symbol})
            
            # Prepare economic data
            for data_type, data in economic_data.items():
                if data:
                    df = pd.DataFrame(data)
                    if 'date' in df.columns and 'value' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date').sort_index()
                        df['value'] = pd.to_numeric(df['value'], errors='coerce')
                        prepared_data[f"economic_{data_type}"] = df[['value']].rename(columns={'value': data_type})
            
            return prepared_data
            
        except Exception as e:
            self.logger.error(f"Error preparing correlation data: {e}")
            return {}
    
    def analyze_rates_vs_stocks(
        self, 
        stock_data: List[Dict[str, Any]], 
        interest_rate_data: List[Dict[str, Any]],
        stock_symbol: str = "market"
    ) -> Dict[str, Any]:
        """
        Analyze correlation between interest rates and stock prices.
        
        Args:
            stock_data: Stock price data
            interest_rate_data: Interest rate data
            stock_symbol: Symbol for labeling
            
        Returns:
            Correlation analysis results
        """
        try:
            # Prepare DataFrames
            stock_df = pd.DataFrame(stock_data)
            rate_df = pd.DataFrame(interest_rate_data)
            
            if stock_df.empty or rate_df.empty:
                return {"error": "Insufficient data for correlation analysis"}
            
            # Align dates and merge data
            stock_df['date'] = pd.to_datetime(stock_df['date'])
            rate_df['date'] = pd.to_datetime(rate_df['date'])
            
            stock_df = stock_df.set_index('date')
            rate_df = rate_df.set_index('date')
            
            # Merge on common dates
            merged_df = pd.merge(
                stock_df[['close_price']], 
                rate_df[['value']], 
                left_index=True, 
                right_index=True, 
                how='inner'
            )
            
            if len(merged_df) < 10:  # Need at least 10 data points
                return {"error": "Insufficient overlapping data points"}
            
            # Calculate correlation
            correlation = merged_df['close_price'].corr(merged_df['value'])
            
            # Calculate rolling correlations for trend analysis
            window_size = min(30, len(merged_df) // 2)
            if window_size >= 5:
                rolling_corr = merged_df['close_price'].rolling(window=window_size).corr(
                    merged_df['value'].rolling(window=window_size)
                )
                recent_correlation = rolling_corr.iloc[-1] if not pd.isna(rolling_corr.iloc[-1]) else correlation
            else:
                recent_correlation = correlation
            
            # Interpret correlation strength
            abs_corr = abs(correlation)
            if abs_corr > 0.7:
                strength = "strong"
            elif abs_corr > 0.4:
                strength = "moderate"
            elif abs_corr > 0.2:
                strength = "weak"
            else:
                strength = "very weak"
            
            # Determine relationship direction
            if correlation > 0:
                direction = "positive"
                interpretation = "Stock prices tend to move in the same direction as interest rates"
            elif correlation < 0:
                direction = "negative"
                interpretation = "Stock prices tend to move opposite to interest rates"
            else:
                direction = "neutral"
                interpretation = "No clear relationship between stock prices and interest rates"
            
            return {
                "correlation": float(correlation),
                "recent_correlation": float(recent_correlation),
                "strength": strength,
                "direction": direction,
                "interpretation": interpretation,
                "data_points": len(merged_df),
                "analysis_period": {
                    "start": merged_df.index.min().isoformat(),
                    "end": merged_df.index.max().isoformat()
                },
                "stock_symbol": stock_symbol
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing rates vs stocks correlation: {e}")
            return {"error": str(e)}
    
    def analyze_currency_impact(
        self, 
        stock_data: List[Dict[str, Any]], 
        currency_data: List[Dict[str, Any]],
        currency_pair: str = "SEK/USD"
    ) -> Dict[str, Any]:
        """
        Analyze correlation between currency exchange rates and stock performance.
        
        Args:
            stock_data: Stock price data
            currency_data: Currency exchange rate data
            currency_pair: Currency pair description
            
        Returns:
            Currency impact analysis results
        """
        try:
            # Similar to rates_vs_stocks but with currency interpretation
            stock_df = pd.DataFrame(stock_data)
            fx_df = pd.DataFrame(currency_data)
            
            if stock_df.empty or fx_df.empty:
                return {"error": "Insufficient data for currency impact analysis"}
            
            # Prepare and merge data
            stock_df['date'] = pd.to_datetime(stock_df['date'])
            fx_df['date'] = pd.to_datetime(fx_df['date'])
            
            stock_df = stock_df.set_index('date')
            fx_df = fx_df.set_index('date')
            
            merged_df = pd.merge(
                stock_df[['close_price']], 
                fx_df[['value']], 
                left_index=True, 
                right_index=True, 
                how='inner'
            )
            
            if len(merged_df) < 10:
                return {"error": "Insufficient overlapping data points"}
            
            # Calculate correlation
            correlation = merged_df['close_price'].corr(merged_df['value'])
            
            # Calculate percentage changes for impact analysis
            stock_returns = merged_df['close_price'].pct_change()
            fx_returns = merged_df['value'].pct_change()
            
            # Beta coefficient (sensitivity of stocks to FX changes)
            if fx_returns.std() > 0:
                beta = stock_returns.cov(fx_returns) / fx_returns.var()
            else:
                beta = 0
            
            # Impact interpretation for SEK specifically
            if "SEK" in currency_pair and "USD" in currency_pair:
                if correlation > 0:
                    fx_impact = "Weakening SEK (higher SEK/USD) tends to boost stock prices"
                elif correlation < 0:
                    fx_impact = "Strengthening SEK (lower SEK/USD) tends to boost stock prices"
                else:
                    fx_impact = "Currency fluctuations have minimal impact on stock prices"
            else:
                fx_impact = f"Currency correlation: {correlation:.3f}"
            
            return {
                "correlation": float(correlation),
                "beta": float(beta),
                "currency_pair": currency_pair,
                "impact_interpretation": fx_impact,
                "data_points": len(merged_df),
                "avg_fx_volatility": float(fx_returns.std() * np.sqrt(252)),  # Annualized
                "avg_stock_volatility": float(stock_returns.std() * np.sqrt(252)),
                "analysis_period": {
                    "start": merged_df.index.min().isoformat(),
                    "end": merged_df.index.max().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing currency impact: {e}")
            return {"error": str(e)}
    
    def economic_indicator_correlation(
        self, 
        stock_data: List[Dict[str, Any]], 
        housing_data: List[Dict[str, Any]], 
        interest_rate_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze correlation between stocks, housing, and interest rates.
        
        Args:
            stock_data: Stock price data
            housing_data: Housing price index data
            interest_rate_data: Interest rate data
            
        Returns:
            Multi-factor correlation analysis
        """
        try:
            # Prepare all datasets
            datasets = {
                "stocks": stock_data,
                "housing": housing_data,
                "rates": interest_rate_data
            }
            
            prepared_dfs = {}
            for name, data in datasets.items():
                if data:
                    df = pd.DataFrame(data)
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                        
                        if name == "stocks" and 'close_price' in df.columns:
                            prepared_dfs[name] = df[['close_price']].rename(columns={'close_price': name})
                        elif 'value' in df.columns:
                            prepared_dfs[name] = df[['value']].rename(columns={'value': name})
            
            # Merge all data on common dates
            if len(prepared_dfs) < 2:
                return {"error": "Need at least 2 datasets for correlation analysis"}
            
            merged_df = None
            for name, df in prepared_dfs.items():
                if merged_df is None:
                    merged_df = df
                else:
                    merged_df = pd.merge(merged_df, df, left_index=True, right_index=True, how='inner')
            
            if merged_df.empty or len(merged_df) < 10:
                return {"error": "Insufficient overlapping data"}
            
            # Calculate correlation matrix
            correlation_matrix = merged_df.corr()
            
            # Extract specific correlations
            correlations = {}
            for col1 in correlation_matrix.columns:
                for col2 in correlation_matrix.columns:
                    if col1 != col2:
                        key = f"{col1}_vs_{col2}"
                        correlations[key] = float(correlation_matrix.loc[col1, col2])
            
            # Calculate returns correlations
            returns_df = merged_df.pct_change().dropna()
            returns_correlation = returns_df.corr() if len(returns_df) > 5 else correlation_matrix
            
            # Interpretation
            interpretations = {}
            if "stocks" in merged_df.columns and "housing" in merged_df.columns:
                stock_housing_corr = correlation_matrix.loc["stocks", "housing"]
                if stock_housing_corr > 0.5:
                    interpretations["stocks_housing"] = "Strong positive correlation - both markets move together"
                elif stock_housing_corr < -0.5:
                    interpretations["stocks_housing"] = "Strong negative correlation - markets move in opposite directions"
                else:
                    interpretations["stocks_housing"] = "Moderate correlation between stock and housing markets"
            
            if "stocks" in merged_df.columns and "rates" in merged_df.columns:
                stock_rates_corr = correlation_matrix.loc["stocks", "rates"]
                if stock_rates_corr < -0.3:
                    interpretations["stocks_rates"] = "Rising rates tend to pressure stock prices"
                elif stock_rates_corr > 0.3:
                    interpretations["stocks_rates"] = "Rising rates tend to support stock prices"
                else:
                    interpretations["stocks_rates"] = "Interest rates have limited impact on stocks"
            
            return {
                "correlation_matrix": correlation_matrix.to_dict(),
                "returns_correlation": returns_correlation.to_dict(),
                "individual_correlations": correlations,
                "interpretations": interpretations,
                "data_points": len(merged_df),
                "analysis_period": {
                    "start": merged_df.index.min().isoformat(),
                    "end": merged_df.index.max().isoformat()
                },
                "datasets_analyzed": list(prepared_dfs.keys())
            }
            
        except Exception as e:
            self.logger.error(f"Error in economic indicator correlation analysis: {e}")
            return {"error": str(e)}
    
    def calculate_portfolio_correlations(
        self, 
        portfolio_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Calculate correlations within a portfolio of assets.
        
        Args:
            portfolio_data: Dictionary of asset data by symbol
            
        Returns:
            Portfolio correlation analysis
        """
        try:
            # Prepare price data for each asset
            price_data = {}
            for symbol, data in portfolio_data.items():
                if data:
                    df = pd.DataFrame(data)
                    if 'date' in df.columns and 'close_price' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date').sort_index()
                        price_data[symbol] = df['close_price']
            
            if len(price_data) < 2:
                return {"error": "Need at least 2 assets for portfolio correlation"}
            
            # Create combined DataFrame
            portfolio_df = pd.DataFrame(price_data)
            portfolio_df = portfolio_df.dropna()
            
            if len(portfolio_df) < 10:
                return {"error": "Insufficient data for portfolio correlation"}
            
            # Calculate price correlations
            price_correlations = portfolio_df.corr()
            
            # Calculate return correlations
            returns_df = portfolio_df.pct_change().dropna()
            returns_correlations = returns_df.corr()
            
            # Calculate diversification metrics
            avg_correlation = returns_correlations.values[np.triu_indices_from(returns_correlations.values, k=1)].mean()
            max_correlation = returns_correlations.values[np.triu_indices_from(returns_correlations.values, k=1)].max()
            min_correlation = returns_correlations.values[np.triu_indices_from(returns_correlations.values, k=1)].min()
            
            # Diversification score (lower correlation = better diversification)
            diversification_score = 1 - avg_correlation
            
            if diversification_score > 0.7:
                diversification_level = "excellent"
            elif diversification_score > 0.5:
                diversification_level = "good"
            elif diversification_score > 0.3:
                diversification_level = "moderate"
            else:
                diversification_level = "poor"
            
            return {
                "price_correlations": price_correlations.to_dict(),
                "returns_correlations": returns_correlations.to_dict(),
                "diversification_metrics": {
                    "average_correlation": float(avg_correlation),
                    "max_correlation": float(max_correlation),
                    "min_correlation": float(min_correlation),
                    "diversification_score": float(diversification_score),
                    "diversification_level": diversification_level
                },
                "portfolio_assets": list(portfolio_data.keys()),
                "data_points": len(portfolio_df),
                "analysis_period": {
                    "start": portfolio_df.index.min().isoformat(),
                    "end": portfolio_df.index.max().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio correlations: {e}")
            return {"error": str(e)}