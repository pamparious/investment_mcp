"""YFinance data collector for historical fund data."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.config.settings import get_settings, FUND_TICKER_MAPPING, ALL_FUNDS
from ...models.fund import Fund, FundPrice
from ...utils.data_quality import DataQualityAssessment
from ...utils.exceptions import DataCollectionError


logger = logging.getLogger(__name__)


class YFinanceCollector:
    """Collector for historical fund data using YFinance."""
    
    def __init__(self):
        """Initialize the YFinance collector."""
        self.settings = get_settings()
        self.data_quality = DataQualityAssessment()
        
    async def collect_historical_data(
        self,
        fund_names: Optional[List[str]] = None,
        years_back: int = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        session: Optional[AsyncSession] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect historical data for specified funds.
        
        Args:
            fund_names: List of fund names to collect data for
            years_back: Number of years of historical data
            start_date: Start date for data collection
            end_date: End date for data collection
            session: Database session for storing data
            
        Returns:
            Dictionary mapping fund names to DataFrames with historical data
        """
        if fund_names is None:
            fund_names = ALL_FUNDS
        
        if years_back is None:
            years_back = self.settings.historical_years
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=years_back * 365)
        
        if end_date is None:
            end_date = datetime.now()
        
        logger.info(
            f"Collecting historical data for {len(fund_names)} funds "
            f"from {start_date.date()} to {end_date.date()}"
        )
        
        results = {}
        
        # Collect data for each fund
        for fund_name in fund_names:
            try:
                logger.info(f"Collecting data for {fund_name}")
                
                data = await self._collect_fund_data(
                    fund_name, start_date, end_date
                )
                
                if data is not None and not data.empty:
                    # Assess data quality
                    quality_score = self.data_quality.assess_data_quality(data)
                    data['data_quality_score'] = quality_score
                    
                    results[fund_name] = data
                    
                    # Store in database if session provided
                    if session is not None:
                        await self._store_fund_data(session, fund_name, data)
                    
                    logger.info(
                        f"Collected {len(data)} records for {fund_name} "
                        f"(quality: {quality_score:.2f})"
                    )
                else:
                    logger.warning(f"No data collected for {fund_name}")
                    
            except Exception as e:
                logger.error(f"Failed to collect data for {fund_name}: {e}")
                continue
        
        logger.info(f"Data collection completed for {len(results)}/{len(fund_names)} funds")
        return results
    
    async def _collect_fund_data(
        self,
        fund_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Collect data for a single fund."""
        
        # Get ticker symbol
        ticker = FUND_TICKER_MAPPING.get(fund_name)
        if not ticker:
            logger.error(f"No ticker mapping found for {fund_name}")
            return None
        
        try:
            # Download data from YFinance
            yf_ticker = yf.Ticker(ticker)
            data = yf_ticker.history(
                start=start_date,
                end=end_date,
                auto_adjust=True,
                prepost=True,
                threads=True
            )
            
            if data.empty:
                logger.warning(f"No data returned from YFinance for {ticker}")
                return None
            
            # Process and clean the data
            processed_data = self._process_raw_data(data, fund_name, ticker)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error collecting data from YFinance for {ticker}: {e}")
            return None
    
    def _process_raw_data(
        self,
        raw_data: pd.DataFrame,
        fund_name: str,
        ticker: str
    ) -> pd.DataFrame:
        """Process raw YFinance data."""
        
        df = raw_data.copy()
        
        # Ensure we have required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Missing column {col} for {fund_name}")
                df[col] = None
        
        # Clean data
        df = df.dropna(subset=['Close'])
        df = df[df['Close'] > 0]
        
        # Calculate returns
        df['daily_return'] = df['Close'].pct_change()
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Calculate moving averages
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['sma_200'] = df['Close'].rolling(window=200).mean()
        
        # Calculate exponential moving averages
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # Calculate volatility measures
        df['volatility_30d'] = df['daily_return'].rolling(window=30).std() * np.sqrt(252)
        df['volatility_90d'] = df['daily_return'].rolling(window=90).std() * np.sqrt(252)
        df['volatility_252d'] = df['daily_return'].rolling(window=252).std() * np.sqrt(252)
        
        # Calculate technical indicators
        df['rsi'] = self._calculate_rsi(df['Close'])
        df['macd'], df['macd_signal'] = self._calculate_macd(df['Close'])
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Calculate Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['Close'].rolling(bb_period).mean()
        bb_rolling_std = df['Close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_rolling_std * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_rolling_std * bb_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Calculate drawdown
        df['cumulative_return'] = (1 + df['daily_return'].fillna(0)).cumprod()
        df['running_max'] = df['cumulative_return'].expanding().max()
        df['drawdown'] = (df['cumulative_return'] - df['running_max']) / df['running_max']
        
        # Calculate rolling Sharpe ratio
        risk_free_rate = self.settings.risk_free_rate
        excess_returns = df['daily_return'] - (risk_free_rate / 252)
        df['sharpe_252d'] = (
            excess_returns.rolling(252).mean() * 252
        ) / (df['daily_return'].rolling(252).std() * np.sqrt(252))
        
        # Calculate Value at Risk
        df['var_5_30d'] = df['daily_return'].rolling(30).quantile(0.05)
        df['var_5_90d'] = df['daily_return'].rolling(90).quantile(0.05)
        
        # Market regime indicators
        df['is_bull_market'] = (df['Close'] > df['sma_200']) & (df['sma_50'] > df['sma_200'])
        df['is_bear_market'] = (df['Close'] < df['sma_200']) & (df['sma_50'] < df['sma_200'])
        df['trend_strength'] = (df['Close'] - df['sma_200']) / df['sma_200']
        
        # Add metadata
        df['fund_name'] = fund_name
        df['ticker'] = ticker
        df['data_source'] = 'yfinance'
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    async def _store_fund_data(
        self,
        session: AsyncSession,
        fund_name: str,
        data: pd.DataFrame
    ) -> None:
        """Store fund data in the database."""
        
        try:
            # Get or create fund record
            fund = await self._get_or_create_fund(session, fund_name)
            
            # Store price data
            await self._store_price_data(session, fund.id, data)
            
            # Store analytics data
            await self._store_analytics_data(session, fund.id, data)
            
            await session.commit()
            
        except Exception as e:
            logger.error(f"Error storing data for {fund_name}: {e}")
            await session.rollback()
            raise
    
    async def _get_or_create_fund(
        self,
        session: AsyncSession,
        fund_name: str
    ) -> Fund:
        """Get existing fund or create new one."""
        
        from sqlalchemy import select
        
        # Try to get existing fund
        result = await session.execute(
            select(Fund).where(Fund.name == fund_name)
        )
        fund = result.scalar_one_or_none()
        
        if fund is None:
            # Create new fund
            from ...core.config.settings import get_fund_category, get_fund_ticker
            
            fund = Fund(
                name=fund_name,
                ticker=get_fund_ticker(fund_name),
                category=get_fund_category(fund_name) or "unknown",
                currency="SEK",
                is_active=True
            )
            session.add(fund)
            await session.flush()  # Get the ID
        
        return fund
    
    async def _store_price_data(
        self,
        session: AsyncSession,
        fund_id: str,
        data: pd.DataFrame
    ) -> None:
        """Store price data in the database."""
        
        from sqlalchemy import select
        
        # Get existing dates to avoid duplicates
        result = await session.execute(
            select(FundPrice.date).where(FundPrice.fund_id == fund_id)
        )
        existing_dates = {row[0] for row in result.fetchall()}
        
        # Prepare price records
        price_records = []
        for date, row in data.iterrows():
            if date.date() not in existing_dates:
                price_record = FundPrice(
                    fund_id=fund_id,
                    date=date.date(),
                    open_price=row.get('Open'),
                    high_price=row.get('High'),
                    low_price=row.get('Low'),
                    close_price=row['Close'],
                    adjusted_close=row.get('Adj Close'),
                    volume=row.get('Volume'),
                    daily_return=row.get('daily_return'),
                    data_source='yfinance',
                    data_quality=self._determine_data_quality(row)
                )
                price_records.append(price_record)
        
        # Bulk insert
        if price_records:
            session.add_all(price_records)
            logger.info(f"Added {len(price_records)} price records for fund {fund_id}")
    
    async def _store_analytics_data(
        self,
        session: AsyncSession,
        fund_id: str,
        data: pd.DataFrame
    ) -> None:
        """Store analytics data in the database."""
        
        from ...models.fund import FundAnalytics
        from sqlalchemy import select
        
        # Get existing dates to avoid duplicates
        result = await session.execute(
            select(FundAnalytics.date).where(FundAnalytics.fund_id == fund_id)
        )
        existing_dates = {row[0] for row in result.fetchall()}
        
        # Prepare analytics records
        analytics_records = []
        for date, row in data.iterrows():
            if date.date() not in existing_dates:
                analytics_record = FundAnalytics(
                    fund_id=fund_id,
                    date=date.date(),
                    sma_20=row.get('sma_20'),
                    sma_50=row.get('sma_50'),
                    sma_200=row.get('sma_200'),
                    ema_12=row.get('ema_12'),
                    ema_26=row.get('ema_26'),
                    rsi=row.get('rsi'),
                    macd=row.get('macd'),
                    macd_signal=row.get('macd_signal'),
                    macd_histogram=row.get('macd_histogram'),
                    bollinger_upper=row.get('bb_upper'),
                    bollinger_lower=row.get('bb_lower'),
                    bollinger_width=row.get('bb_width'),
                    volatility_30d=row.get('volatility_30d'),
                    volatility_90d=row.get('volatility_90d'),
                    volatility_252d=row.get('volatility_252d'),
                    sharpe_252d=row.get('sharpe_252d'),
                    var_5_30d=row.get('var_5_30d'),
                    var_5_90d=row.get('var_5_90d'),
                    max_drawdown_252d=row.get('drawdown'),
                    is_bull_market=row.get('is_bull_market'),
                    is_bear_market=row.get('is_bear_market'),
                    trend_strength=row.get('trend_strength')
                )
                analytics_records.append(analytics_record)
        
        # Bulk insert
        if analytics_records:
            session.add_all(analytics_records)
            logger.info(f"Added {len(analytics_records)} analytics records for fund {fund_id}")
    
    def _determine_data_quality(self, row: pd.Series) -> str:
        """Determine data quality for a single row."""
        
        # Check for missing values
        required_fields = ['Open', 'High', 'Low', 'Close']
        missing_count = sum(1 for field in required_fields if pd.isna(row.get(field)))
        
        if missing_count == 0:
            return "high"
        elif missing_count <= 2:
            return "medium"
        else:
            return "low"
    
    async def update_recent_data(
        self,
        fund_names: Optional[List[str]] = None,
        days_back: int = 5,
        session: Optional[AsyncSession] = None
    ) -> Dict[str, pd.DataFrame]:
        """Update recent data for funds."""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        return await self.collect_historical_data(
            fund_names=fund_names,
            start_date=start_date,
            end_date=end_date,
            session=session
        )
    
    async def validate_data_coverage(
        self,
        fund_names: Optional[List[str]] = None,
        required_years: int = None
    ) -> Dict[str, Dict[str, any]]:
        """Validate data coverage for funds."""
        
        if fund_names is None:
            fund_names = ALL_FUNDS
        
        if required_years is None:
            required_years = self.settings.historical_years
        
        results = {}
        
        for fund_name in fund_names:
            ticker = FUND_TICKER_MAPPING.get(fund_name)
            if not ticker:
                results[fund_name] = {
                    "status": "error",
                    "message": "No ticker mapping found"
                }
                continue
            
            try:
                # Get basic info about data availability
                yf_ticker = yf.Ticker(ticker)
                info = yf_ticker.info
                
                # Try to get some historical data to check availability
                test_end = datetime.now()
                test_start = test_end - timedelta(days=required_years * 365)
                
                test_data = yf_ticker.history(
                    start=test_start,
                    end=test_end,
                    auto_adjust=True
                )
                
                if not test_data.empty:
                    first_date = test_data.index[0]
                    last_date = test_data.index[-1]
                    data_years = (last_date - first_date).days / 365.25
                    
                    results[fund_name] = {
                        "status": "available",
                        "first_date": first_date.date(),
                        "last_date": last_date.date(),
                        "data_years": data_years,
                        "data_points": len(test_data),
                        "meets_requirement": data_years >= required_years
                    }
                else:
                    results[fund_name] = {
                        "status": "no_data",
                        "message": "No historical data available"
                    }
                    
            except Exception as e:
                results[fund_name] = {
                    "status": "error",
                    "message": str(e)
                }
        
        return results