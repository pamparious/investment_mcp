"""Collect 20+ years of historical data for the 12 specific funds."""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
import logging
import pickle
import os

logger = logging.getLogger(__name__)


class HistoricalFundCollector:
    """Collect 20+ years of historical data for the 12 specific funds."""
    
    # Mapping of our fund codes to actual tickers/identifiers
    FUND_TICKERS = {
        "DNB_GLOBAL_INDEKS_S": "^GSPC",  # Use S&P 500 as proxy for global equity
        "AVANZA_EMERGING_MARKETS": "EEM", # iShares MSCI Emerging Markets ETF
        "STOREBRAND_EUROPA_A_SEK": "VGK", # Vanguard European ETF
        "DNB_NORDEN_INDEKS_S": "^OMX",   # OMX Stockholm as Nordic proxy
        "PLUS_ALLABOLAG_SVERIGE_INDEX": "^OMX", # OMX Stockholm for Swedish equity
        "AVANZA_USA": "^GSPC",           # S&P 500 for US equity
        "STOREBRAND_JAPAN_A_SEK": "EWJ", # iShares MSCI Japan ETF
        "HANDELSBANKEN_GLOBAL_SMAB_INDEX": "IWM", # Russell 2000 for small cap
        "XETRA_GOLD_ETC": "GLD",         # SPDR Gold Trust ETF
        "VIRTUNE_BITCOIN_PRIME_ETP": "BTC-USD", # Bitcoin
        "XBT_ETHER_ONE": "ETH-USD",      # Ethereum
        "PLUS_FASTIGHETER_SVERIGE_INDEX": "VNQ"  # Vanguard Real Estate ETF
    }
    
    # Alternative/secondary data sources
    PROXY_INDICES = {
        "DNB_GLOBAL_INDEKS_S": ["^GSPC", "ACWI", "VTI"],
        "AVANZA_EMERGING_MARKETS": ["EEM", "VWO", "IEMG"],
        "STOREBRAND_EUROPA_A_SEK": ["VGK", "EZU", "IEUR"],
        "DNB_NORDEN_INDEKS_S": ["^OMX", "^GSPC", "EWD"],
        "PLUS_ALLABOLAG_SVERIGE_INDEX": ["^OMX", "EWD", "^GSPC"],
        "AVANZA_USA": ["^GSPC", "VTI", "ITOT"],
        "STOREBRAND_JAPAN_A_SEK": ["EWJ", "DXJ", "^N225"],
        "HANDELSBANKEN_GLOBAL_SMAB_INDEX": ["IWM", "VB", "IJR"],
        "XETRA_GOLD_ETC": ["GLD", "IAU", "SGOL"],
        "VIRTUNE_BITCOIN_PRIME_ETP": ["BTC-USD"],
        "XBT_ETHER_ONE": ["ETH-USD"],
        "PLUS_FASTIGHETER_SVERIGE_INDEX": ["VNQ", "IYR", "XLRE"]
    }
    
    def __init__(self):
        """Initialize the historical fund collector."""
        self.cache_dir = "data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
    async def collect_all_historical_data(self, years_back: int = 20) -> Dict[str, pd.DataFrame]:
        """Collect historical data for all 12 funds."""
        
        logger.info(f"Collecting historical data for all 12 funds ({years_back} years)")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)
        
        results = {}
        
        for fund_code in self.FUND_TICKERS.keys():
            try:
                logger.info(f"Collecting data for {fund_code}")
                
                # Check cache first
                cached_data = self.load_cached_data(fund_code)
                if cached_data is not None and self.is_cache_valid(fund_code):
                    logger.info(f"Using cached data for {fund_code}")
                    results[fund_code] = cached_data
                    continue
                
                # Collect fresh data
                data = await self.get_fund_historical_data(fund_code, start_date, end_date)
                results[fund_code] = data
                
                # Cache the data
                self.save_cached_data(data, fund_code)
                
                # Rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to collect data for {fund_code}: {e}")
                results[fund_code] = None
        
        logger.info(f"Completed data collection for {len([r for r in results.values() if r is not None])}/{len(results)} funds")
        return results
    
    async def get_fund_historical_data(self, fund_code: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get historical data for a specific fund."""
        
        tickers_to_try = self.PROXY_INDICES.get(fund_code, [self.FUND_TICKERS.get(fund_code)])
        
        for ticker in tickers_to_try:
            try:
                logger.info(f"Trying ticker {ticker} for {fund_code}")
                
                # Special handling for cryptocurrencies
                if ticker in ["BTC-USD", "ETH-USD"]:
                    data = await self.get_crypto_historical_data(ticker, fund_code, start_date, end_date)
                else:
                    data = await self.get_traditional_asset_data(ticker, fund_code, start_date, end_date)
                
                if len(data) > 100:  # Sufficient data
                    return data
                    
            except Exception as e:
                logger.warning(f"Failed to get data from {ticker} for {fund_code}: {e}")
                continue
        
        # If all sources fail, generate synthetic data based on asset class
        logger.warning(f"No data sources worked for {fund_code}, generating synthetic data")
        return self.generate_synthetic_data(fund_code, start_date, end_date)
    
    async def get_traditional_asset_data(self, ticker: str, fund_code: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get data for traditional assets (stocks, ETFs, indices)."""
        
        try:
            yf_ticker = yf.Ticker(ticker)
            data = yf_ticker.history(start=start_date, end=end_date, auto_adjust=True, prepost=True)
            
            if data.empty:
                raise ValueError(f"No data returned for {ticker}")
            
            return self.process_fund_data(data, fund_code, ticker, is_proxy=ticker != self.FUND_TICKERS.get(fund_code))
            
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {ticker}: {e}")
    
    async def get_crypto_historical_data(self, symbol: str, fund_code: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Special handling for cryptocurrency data."""
        
        try:
            # Bitcoin has data from ~2010, Ethereum from ~2015
            if symbol == "BTC-USD":
                min_start = datetime(2010, 7, 1)
            elif symbol == "ETH-USD":
                min_start = datetime(2015, 8, 1)
            else:
                min_start = start_date
            
            effective_start = max(min_start, start_date)
            
            yf_ticker = yf.Ticker(symbol)
            data = yf_ticker.history(start=effective_start, end=end_date, auto_adjust=True)
            
            if data.empty:
                raise ValueError(f"No crypto data for {symbol}")
            
            return self.process_fund_data(data, fund_code, symbol, is_crypto=True)
            
        except Exception as e:
            raise ValueError(f"Failed to fetch crypto data for {symbol}: {e}")
    
    def process_fund_data(self, data: pd.DataFrame, fund_code: str, ticker: str, is_proxy: bool = False, is_crypto: bool = False) -> pd.DataFrame:
        """Process and enhance fund data with technical indicators."""
        
        df = data.copy()
        
        # Ensure we have the basic columns
        if 'Close' not in df.columns:
            raise ValueError("Missing Close price data")
        
        # Clean data
        df = df.dropna(subset=['Close'])
        df = df[df['Close'] > 0]  # Remove invalid prices
        
        # Calculate returns
        df['daily_return'] = df['Close'].pct_change()
        df['monthly_return'] = df['Close'].pct_change(periods=21)  # ~21 trading days
        df['quarterly_return'] = df['Close'].pct_change(periods=63)  # ~3 months
        df['annual_return'] = df['Close'].pct_change(periods=252)  # ~1 year
        
        # Calculate volatility (rolling windows)
        df['volatility_30d'] = df['daily_return'].rolling(30).std() * np.sqrt(252)
        df['volatility_90d'] = df['daily_return'].rolling(90).std() * np.sqrt(252)
        df['volatility_252d'] = df['daily_return'].rolling(252).std() * np.sqrt(252)
        
        # Calculate moving averages
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['sma_200'] = df['Close'].rolling(200).mean()
        
        # Calculate exponential moving averages
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # Calculate technical indicators
        df['rsi'] = self.calculate_rsi(df['Close'])
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Calculate Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['Close'].rolling(bb_period).mean()
        bb_rolling_std = df['Close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_rolling_std * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_rolling_std * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Calculate drawdown metrics
        df['cumulative_return'] = (1 + df['daily_return'].fillna(0)).cumprod()
        df['running_max'] = df['cumulative_return'].expanding().max()
        df['drawdown'] = (df['cumulative_return'] - df['running_max']) / df['running_max']
        
        # Calculate rolling Sharpe ratio (252-day)
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        excess_returns = df['daily_return'] - (risk_free_rate / 252)
        df['sharpe_252d'] = (excess_returns.rolling(252).mean() * 252) / (df['daily_return'].rolling(252).std() * np.sqrt(252))
        
        # Calculate Value at Risk (5%)
        df['var_5_30d'] = df['daily_return'].rolling(30).quantile(0.05)
        df['var_5_90d'] = df['daily_return'].rolling(90).quantile(0.05)
        
        # Calculate skewness and kurtosis
        df['skewness_90d'] = df['daily_return'].rolling(90).skew()
        df['kurtosis_90d'] = df['daily_return'].rolling(90).kurt()
        
        # Market regime indicators
        df['bull_market'] = (df['Close'] > df['sma_200']) & (df['sma_50'] > df['sma_200'])
        df['bear_market'] = (df['Close'] < df['sma_200']) & (df['sma_50'] < df['sma_200'])
        
        # Trend strength
        df['trend_strength'] = (df['Close'] - df['sma_200']) / df['sma_200']
        
        # Add metadata
        df['fund_code'] = fund_code
        df['source_ticker'] = ticker
        df['is_proxy_data'] = is_proxy
        df['is_crypto'] = is_crypto
        df['data_quality'] = self.assess_data_quality(df)
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def assess_data_quality(self, df: pd.DataFrame) -> str:
        """Assess the quality of the data."""
        
        total_points = len(df)
        missing_points = df['Close'].isna().sum()
        coverage = 1 - (missing_points / total_points)
        
        # Check for large gaps
        date_diffs = df.index.to_series().diff().dt.days
        large_gaps = (date_diffs > 7).sum()  # More than 7 days gap
        
        # Check for extreme outliers
        returns = df['daily_return'].dropna()
        extreme_moves = ((returns.abs() > 0.15).sum() / len(returns)) if len(returns) > 0 else 0
        
        if coverage > 0.95 and large_gaps < 5 and extreme_moves < 0.01:
            return "high"
        elif coverage > 0.90 and large_gaps < 20 and extreme_moves < 0.02:
            return "medium"
        else:
            return "low"
    
    def generate_synthetic_data(self, fund_code: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate synthetic data based on asset class characteristics."""
        
        logger.warning(f"Generating synthetic data for {fund_code}")
        
        # Asset class characteristics
        asset_characteristics = {
            "DNB_GLOBAL_INDEKS_S": {"annual_return": 0.08, "volatility": 0.16},
            "AVANZA_EMERGING_MARKETS": {"annual_return": 0.06, "volatility": 0.22},
            "STOREBRAND_EUROPA_A_SEK": {"annual_return": 0.07, "volatility": 0.18},
            "DNB_NORDEN_INDEKS_S": {"annual_return": 0.09, "volatility": 0.20},
            "PLUS_ALLABOLAG_SVERIGE_INDEX": {"annual_return": 0.09, "volatility": 0.20},
            "AVANZA_USA": {"annual_return": 0.10, "volatility": 0.16},
            "STOREBRAND_JAPAN_A_SEK": {"annual_return": 0.05, "volatility": 0.19},
            "HANDELSBANKEN_GLOBAL_SMAB_INDEX": {"annual_return": 0.09, "volatility": 0.24},
            "XETRA_GOLD_ETC": {"annual_return": 0.04, "volatility": 0.16},
            "VIRTUNE_BITCOIN_PRIME_ETP": {"annual_return": 0.15, "volatility": 0.80},
            "XBT_ETHER_ONE": {"annual_return": 0.20, "volatility": 0.90},
            "PLUS_FASTIGHETER_SVERIGE_INDEX": {"annual_return": 0.08, "volatility": 0.18}
        }
        
        char = asset_characteristics.get(fund_code, {"annual_return": 0.07, "volatility": 0.18})
        
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate synthetic returns using GBM
        np.random.seed(hash(fund_code) % 2**32)  # Consistent seed for reproducibility
        
        daily_return = char["annual_return"] / 252
        daily_vol = char["volatility"] / np.sqrt(252)
        
        returns = np.random.normal(daily_return, daily_vol, len(dates))
        
        # Add some autocorrelation and regime changes
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]  # Small autocorrelation
        
        # Generate price series
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.005, len(prices))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, len(prices)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(prices)))),
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, len(prices))
        }, index=dates)
        
        # Ensure High >= Low and both contain Close
        df['High'] = df[['High', 'Close', 'Open']].max(axis=1)
        df['Low'] = df[['Low', 'Close', 'Open']].min(axis=1)
        
        # Process the synthetic data the same way
        processed_df = self.process_fund_data(df, fund_code, "SYNTHETIC", is_proxy=True)
        processed_df['is_synthetic'] = True
        
        return processed_df
    
    def load_cached_data(self, fund_code: str) -> Optional[pd.DataFrame]:
        """Load cached data for a fund."""
        cache_file = os.path.join(self.cache_dir, f"{fund_code}_historical.pkl")
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached data for {fund_code}: {e}")
        
        return None
    
    def save_cached_data(self, data: pd.DataFrame, fund_code: str) -> None:
        """Save data to cache."""
        cache_file = os.path.join(self.cache_dir, f"{fund_code}_historical.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Cached data for {fund_code}")
        except Exception as e:
            logger.warning(f"Failed to cache data for {fund_code}: {e}")
    
    def is_cache_valid(self, fund_code: str, max_age_days: int = 1) -> bool:
        """Check if cached data is still valid."""
        cache_file = os.path.join(self.cache_dir, f"{fund_code}_historical.pkl")
        
        if not os.path.exists(cache_file):
            return False
        
        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
        return file_age.days < max_age_days
    
    async def collect_recent_data(self, days: int = 30) -> Dict[str, pd.DataFrame]:
        """Collect recent data for all funds (for daily updates)."""
        
        logger.info(f"Collecting recent {days} days of data")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        results = {}
        
        for fund_code in self.FUND_TICKERS.keys():
            try:
                data = await self.get_fund_historical_data(fund_code, start_date, end_date)
                results[fund_code] = data
                await asyncio.sleep(0.2)  # Faster rate limiting for recent data
                
            except Exception as e:
                logger.error(f"Failed to collect recent data for {fund_code}: {e}")
                results[fund_code] = None
        
        return results
    
    def calculate_total_return(self, data: pd.DataFrame, years: int) -> float:
        """Calculate total return over specified years."""
        if len(data) < years * 252:  # Not enough data
            return np.nan
        
        end_price = data['Close'].iloc[-1]
        start_price = data['Close'].iloc[-(years * 252)]
        
        return (end_price / start_price) - 1
    
    def calculate_worst_year_return(self, data: pd.DataFrame) -> float:
        """Calculate worst calendar year return."""
        if len(data) < 252:
            return np.nan
        
        yearly_returns = []
        current_year = data.index[-1].year
        
        for year in range(data.index[0].year, current_year + 1):
            year_data = data[data.index.year == year]
            if len(year_data) > 50:  # Sufficient data for the year
                year_return = (year_data['Close'].iloc[-1] / year_data['Close'].iloc[0]) - 1
                yearly_returns.append(year_return)
        
        return min(yearly_returns) if yearly_returns else np.nan
    
    def calculate_best_year_return(self, data: pd.DataFrame) -> float:
        """Calculate best calendar year return."""
        if len(data) < 252:
            return np.nan
        
        yearly_returns = []
        current_year = data.index[-1].year
        
        for year in range(data.index[0].year, current_year + 1):
            year_data = data[data.index.year == year]
            if len(year_data) > 50:
                year_return = (year_data['Close'].iloc[-1] / year_data['Close'].iloc[0]) - 1
                yearly_returns.append(year_return)
        
        return max(yearly_returns) if yearly_returns else np.nan
    
    def calculate_positive_years_percentage(self, data: pd.DataFrame) -> float:
        """Calculate percentage of positive years."""
        if len(data) < 252:
            return np.nan
        
        yearly_returns = []
        current_year = data.index[-1].year
        
        for year in range(data.index[0].year, current_year + 1):
            year_data = data[data.index.year == year]
            if len(year_data) > 50:
                year_return = (year_data['Close'].iloc[-1] / year_data['Close'].iloc[0]) - 1
                yearly_returns.append(year_return)
        
        if not yearly_returns:
            return np.nan
        
        positive_years = sum(1 for ret in yearly_returns if ret > 0)
        return positive_years / len(yearly_returns)
    
    def calculate_recession_performance(self, data: pd.DataFrame) -> float:
        """Calculate average performance during recession periods."""
        # Simplified: assume bear markets (below 200-day SMA) are recession proxies
        if 'bear_market' not in data.columns:
            return np.nan
        
        recession_returns = data[data['bear_market']]['daily_return']
        return recession_returns.mean() * 252 if len(recession_returns) > 0 else np.nan
    
    def calculate_inflation_hedge_score(self, data: pd.DataFrame) -> float:
        """Calculate a simple inflation hedge score."""
        # Simplified: calculate correlation with trend (assumption: inflation periods have trends)
        if len(data) < 252:
            return np.nan
        
        # Use rolling correlation with time trend as proxy
        trend = np.arange(len(data))
        returns = data['daily_return'].dropna()
        
        if len(returns) < 100:
            return np.nan
        
        correlation = np.corrcoef(trend[-len(returns):], returns)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0