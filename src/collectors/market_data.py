"""
Unified market data collector for Investment MCP System.

This module consolidates all market data collection functionality from various 
scattered collector files into a single, efficient data collection system.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Optional, Any, Tuple
import logging
from sqlalchemy.orm import Session
from ..core.config import get_settings, TRADEABLE_FUNDS
from ..core.database import MarketData

logger = logging.getLogger(__name__)


class MarketDataCollector:
    """Unified market data collector for all 12 approved funds."""
    
    # Mapping of fund IDs to Yahoo Finance tickers
    FUND_TICKERS = {
        "DNB_GLOBAL_INDEKS_S": "^GSPC",  # S&P 500 as global equity proxy
        "AVANZA_EMERGING_MARKETS": "EEM", # iShares MSCI Emerging Markets ETF
        "STOREBRAND_EUROPA_A_SEK": "VGK", # Vanguard European ETF
        "DNB_NORDEN_INDEKS_S": "^OMX",   # OMX Stockholm 30
        "PLUS_ALLABOLAG_SVERIGE_INDEX": "^OMX", # OMX Stockholm 30
        "AVANZA_USA": "^GSPC",           # S&P 500
        "STOREBRAND_JAPAN_A_SEK": "EWJ", # iShares MSCI Japan ETF
        "HANDELSBANKEN_GLOBAL_SMAB_INDEX": "IWM", # iShares Russell 2000 ETF
        "XETRA_GOLD_ETC": "GLD",         # SPDR Gold Trust ETF
        "VIRTUNE_BITCOIN_PRIME_ETP": "BTC-USD", # Bitcoin
        "XBT_ETHER_ONE": "ETH-USD",      # Ethereum
        "PLUS_FASTIGHETER_SVERIGE_INDEX": "VNQ"  # Vanguard Real Estate ETF
    }
    
    # Fallback tickers for better data coverage
    FALLBACK_TICKERS = {
        "DNB_GLOBAL_INDEKS_S": ["^GSPC", "ACWI", "VTI"],
        "AVANZA_EMERGING_MARKETS": ["EEM", "VWO", "IEMG"],
        "STOREBRAND_EUROPA_A_SEK": ["VGK", "EZU", "IEUR"],
        "DNB_NORDEN_INDEKS_S": ["^OMX", "EWD"],
        "PLUS_ALLABOLAG_SVERIGE_INDEX": ["^OMX", "EWD"],
        "AVANZA_USA": ["^GSPC", "VTI", "ITOT"],
        "STOREBRAND_JAPAN_A_SEK": ["EWJ", "DXJ", "^N225"],
        "HANDELSBANKEN_GLOBAL_SMAB_INDEX": ["IWM", "VB", "IJR"],
        "XETRA_GOLD_ETC": ["GLD", "IAU", "SGOL"],
        "VIRTUNE_BITCOIN_PRIME_ETP": ["BTC-USD"],
        "XBT_ETHER_ONE": ["ETH-USD"],
        "PLUS_FASTIGHETER_SVERIGE_INDEX": ["VNQ", "IYR", "XLRE"]
    }
    
    def __init__(self):
        self.settings = get_settings()
        self.timeout = self.settings.YAHOO_FINANCE_TIMEOUT
    
    async def collect_historical_data(
        self, 
        fund_ids: Optional[List[str]] = None,
        years: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect historical data for specified funds or all approved funds.
        
        Args:
            fund_ids: List of fund IDs to collect. If None, collects all approved funds.
            years: Number of years of historical data. If None, uses config default.
            
        Returns:
            Dictionary mapping fund_id to DataFrame with historical data
        """
        if fund_ids is None:
            fund_ids = list(TRADEABLE_FUNDS.keys())
        
        if years is None:
            years = self.settings.HISTORICAL_DATA_YEARS
        
        start_date = datetime.now() - timedelta(days=years * 365)
        end_date = datetime.now()
        
        logger.info(f"Collecting {years} years of data for {len(fund_ids)} funds")
        
        results = {}
        tasks = []
        
        for fund_id in fund_ids:
            task = self._collect_fund_data(fund_id, start_date, end_date)
            tasks.append(task)
        
        # Execute all data collection tasks concurrently
        fund_data_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for fund_id, result in zip(fund_ids, fund_data_results):
            if isinstance(result, Exception):
                logger.error(f"Failed to collect data for {fund_id}: {result}")
                results[fund_id] = pd.DataFrame()
            else:
                results[fund_id] = result
        
        return results
    
    async def _collect_fund_data(
        self, 
        fund_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Collect data for a single fund with fallback options."""
        
        if fund_id not in self.FUND_TICKERS:
            logger.error(f"No ticker mapping found for fund: {fund_id}")
            return pd.DataFrame()
        
        primary_ticker = self.FUND_TICKERS[fund_id]
        fallback_tickers = self.FALLBACK_TICKERS.get(fund_id, [primary_ticker])
        
        # Try primary ticker first, then fallbacks
        for ticker in [primary_ticker] + fallback_tickers:
            try:
                data = await self._fetch_ticker_data(ticker, start_date, end_date)
                if not data.empty:
                    # Add fund metadata
                    data['fund_id'] = fund_id
                    data['fund_name'] = TRADEABLE_FUNDS[fund_id]['name']
                    data['ticker'] = ticker
                    
                    logger.info(f"Successfully collected {len(data)} records for {fund_id} using {ticker}")
                    return data
                    
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker} for {fund_id}: {e}")
                continue
        
        logger.error(f"All data sources failed for {fund_id}")
        return pd.DataFrame()
    
    async def _fetch_ticker_data(
        self, 
        ticker: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch data from Yahoo Finance for a specific ticker."""
        
        try:
            # Use yfinance to download data
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                auto_adjust=True,
                back_adjust=True
            )
            
            if data.empty:
                logger.warning(f"No data returned for ticker: {ticker}")
                return pd.DataFrame()
            
            # Standardize column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns for {ticker}: {missing_columns}")
                # Fill missing columns with close price or 0 for volume
                for col in missing_columns:
                    if col == 'volume':
                        data[col] = 0
                    else:
                        data[col] = data.get('close', 0)
            
            # Reset index to get date as column
            data.reset_index(inplace=True)
            data['date'] = pd.to_datetime(data['date']).dt.date
            
            # Data quality checks
            data = self._clean_data(data, ticker)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            raise
    
    def _clean_data(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Clean and validate the collected data."""
        
        if data.empty:
            return data
        
        # Remove rows with all NaN values
        data = data.dropna(how='all')
        
        # Remove rows where price columns are all zero or negative
        price_columns = ['open', 'high', 'low', 'close']
        valid_prices = data[price_columns] > 0
        data = data[valid_prices.any(axis=1)]
        
        # Forward fill missing values
        data[price_columns] = data[price_columns].fillna(method='ffill')
        
        # Fill volume with 0 if missing
        if 'volume' in data.columns:
            data['volume'] = data['volume'].fillna(0)
        
        # Data quality check
        valid_records = len(data)
        if valid_records == 0:
            logger.warning(f"No valid records after cleaning for {ticker}")
            return pd.DataFrame()
        
        logger.info(f"Cleaned data for {ticker}: {valid_records} valid records")
        return data
    
    async def collect_recent_data(self, days: int = None) -> Dict[str, pd.DataFrame]:
        """Collect recent data for daily updates."""
        
        if days is None:
            days = self.settings.RECENT_DATA_DAYS
        
        start_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()
        
        logger.info(f"Collecting recent {days} days of data")
        
        return await self.collect_historical_data(
            fund_ids=list(TRADEABLE_FUNDS.keys()),
            years=None  # Use date range instead
        )
    
    def save_to_database(self, data_dict: Dict[str, pd.DataFrame], db_session: Session) -> int:
        """Save collected data to database."""
        
        total_records = 0
        
        for fund_id, data in data_dict.items():
            if data.empty:
                continue
            
            fund_info = TRADEABLE_FUNDS.get(fund_id, {})
            fund_name = fund_info.get('name', fund_id)
            
            for _, row in data.iterrows():
                # Check if record already exists
                existing = db_session.query(MarketData).filter(
                    MarketData.symbol == row.get('ticker', fund_id),
                    MarketData.date == row['date']
                ).first()
                
                if existing:
                    # Update existing record
                    existing.open_price = row['open']
                    existing.high_price = row['high']
                    existing.low_price = row['low']
                    existing.close_price = row['close']
                    existing.volume = int(row.get('volume', 0))
                    existing.adjusted_close = row.get('close', row['close'])
                    existing.updated_at = datetime.utcnow()
                else:
                    # Create new record
                    market_data = MarketData(
                        symbol=row.get('ticker', fund_id),
                        fund_name=fund_name,
                        date=row['date'],
                        open_price=row['open'],
                        high_price=row['high'],
                        low_price=row['low'],
                        close_price=row['close'],
                        volume=int(row.get('volume', 0)),
                        adjusted_close=row.get('close', row['close'])
                    )
                    db_session.add(market_data)
                    total_records += 1
        
        db_session.commit()
        logger.info(f"Saved {total_records} new market data records to database")
        
        return total_records
    
    def get_fund_statistics(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Calculate basic statistics for collected fund data."""
        
        statistics = {}
        
        for fund_id, data in data_dict.items():
            if data.empty:
                statistics[fund_id] = {"status": "no_data", "records": 0}
                continue
            
            # Calculate basic statistics
            price_col = 'close'
            if price_col not in data.columns:
                statistics[fund_id] = {"status": "invalid_data", "records": len(data)}
                continue
            
            prices = data[price_col].dropna()
            if len(prices) < 2:
                statistics[fund_id] = {"status": "insufficient_data", "records": len(data)}
                continue
            
            returns = prices.pct_change().dropna()
            
            stats = {
                "status": "success",
                "records": len(data),
                "date_range": {
                    "start": str(data['date'].min()),
                    "end": str(data['date'].max())
                },
                "price_stats": {
                    "current": float(prices.iloc[-1]),
                    "min": float(prices.min()),
                    "max": float(prices.max()),
                    "mean": float(prices.mean())
                },
                "return_stats": {
                    "total_return": float((prices.iloc[-1] / prices.iloc[0] - 1) * 100),
                    "annualized_return": float(returns.mean() * 252 * 100),
                    "volatility": float(returns.std() * np.sqrt(252) * 100),
                    "max_drawdown": float(self._calculate_max_drawdown(prices) * 100)
                }
            }
            
            statistics[fund_id] = stats
        
        return statistics
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown for a price series."""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()