"""Riksbank API data collection module."""

import aiohttp
import asyncio
import logging
import random
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from config.settings import settings
from backend.database import get_db_session
from backend.models import RiksbankData, DataCollectionLog

logger = logging.getLogger(__name__)

@dataclass
class RiksbankSeries:
    """Riksbank data series configuration."""
    series_id: str
    description: str
    unit: str
    frequency: str  # D, M, Q, A for Daily, Monthly, Quarterly, Annual

class RiksbankCollector:
    """Collector for Riksbank data."""
    
    # Key economic indicators from Riksbank (using actual REST API series)
    # Limited to most important series to respect rate limits
    SERIES_CONFIG = {
        # Primary exchange rates (most important for Swedish economy)
        "sekusdpmi": RiksbankSeries("sekusdpmi", "SEK/USD exchange rate", "SEK per USD", "D"),
        "sekeurpmi": RiksbankSeries("sekeurpmi", "SEK/EUR exchange rate", "SEK per EUR", "D"),
        "seknokpmi": RiksbankSeries("seknokpmi", "SEK/NOK exchange rate", "SEK per NOK", "D"),
        
        # Interest rates (common series names for central banks)
        "REPORATE": RiksbankSeries("REPORATE", "Repo rate", "Percent", "D"),
    }
    
    def __init__(self):
        self.base_url = "https://api.riksbank.se"
        self.session = None
        self.api_key = settings.RIKSBANKEN_API_KEY
        self.max_retries = 3
        self.base_delay = 5  # Base delay for exponential backoff
        self.request_delay = 6  # Minimum 6 seconds between requests to respect rate limits
        
    async def __aenter__(self):
        """Async context manager entry."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Ocp-Apim-Subscription-Key"] = self.api_key
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.REQUEST_TIMEOUT),
            headers=headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _make_request_with_retry(self, url: str, max_retries: int = None) -> Optional[Dict[str, Any]]:
        """
        Make an HTTP request with exponential backoff retry logic.
        
        Args:
            url: The URL to request
            max_retries: Maximum number of retries (defaults to self.max_retries)
            
        Returns:
            Response data or None if failed
        """
        if max_retries is None:
            max_retries = self.max_retries
            
        for attempt in range(max_retries + 1):
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 204:
                        # No Content - this is normal, means no data available for this date
                        logger.debug(f"No data available for URL: {url}")
                        return None
                    elif response.status == 404:
                        # Not Found - also normal, means no data for this date/series
                        logger.debug(f"Series/date not found for URL: {url}")
                        return None
                    elif response.status == 429:
                        # Rate Limited - implement exponential backoff
                        if attempt < max_retries:
                            # Calculate delay with exponential backoff and jitter
                            delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                            logger.warning(f"Rate limited (429) for {url}, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries + 1})")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            logger.error(f"Rate limited (429) for {url}, max retries exhausted")
                            return None
                    else:
                        # Other HTTP errors
                        logger.warning(f"Riksbank API error {response.status} for {url}")
                        if attempt < max_retries:
                            # For other errors, use shorter delay
                            delay = 2 * (attempt + 1) + random.uniform(0, 0.5)
                            await asyncio.sleep(delay)
                            continue
                        return None
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {url} (attempt {attempt + 1}/{max_retries + 1})")
                if attempt < max_retries:
                    delay = 2 * (attempt + 1)
                    await asyncio.sleep(delay)
                    continue
                return None
            except Exception as e:
                logger.warning(f"Error fetching {url} (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    delay = 2 * (attempt + 1)
                    await asyncio.sleep(delay)
                    continue
                return None
        
        return None
    
    async def fetch_series_data(
        self, 
        series_id: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch data for a specific Riksbank series using the new REST API.
        
        Args:
            series_id: Riksbank series identifier (e.g., 'sekusdpmi')
            start_date: Start date for data collection
            end_date: End date for data collection
        
        Returns:
            List of data points
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        data_points = []
        
        # For exchange rates, use the swea/v1 endpoint
        if series_id.lower().endswith('pmi'):
            data_points.extend(await self._fetch_exchange_rate_data(series_id, start_date, end_date))
        else:
            # For other data types, try different endpoint patterns
            data_points.extend(await self._fetch_interest_rate_data(series_id, start_date, end_date))
        
        return data_points
    
    async def _fetch_exchange_rate_data(
        self, 
        series_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch exchange rate data using the swea/v1 endpoint."""
        data_points = []
        
        # Limit date range to avoid too many requests (max 7 days for recent data)
        max_days = 7
        actual_start_date = max(start_date, end_date - timedelta(days=max_days))
        
        logger.info(f"Fetching {series_id} data from {actual_start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Generate date range
        current_date = actual_start_date
        request_count = 0
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Construct URL based on the documented pattern
            url = f"{self.base_url}/swea/v1/Observations/{series_id}/{date_str}"
            
            # Use the new retry method
            data = await self._make_request_with_retry(url)
            
            if data and isinstance(data, list) and len(data) > 0:
                # Parse the response data
                for observation in data:
                    if isinstance(observation, dict):
                        data_point = self._parse_observation(observation, series_id)
                        if data_point:
                            data_points.append(data_point)
                            logger.debug(f"Successfully parsed data for {series_id} on {date_str}: {data_point['value']}")
                    else:
                        logger.debug(f"Skipping non-dict observation: {observation}")
            
            # Move to next day
            current_date += timedelta(days=1)
            request_count += 1
            
            # Respect rate limits with progressive delay
            # Add jitter to avoid thundering herd if multiple series are collected
            base_delay = self.request_delay + random.uniform(0, 2)
            await asyncio.sleep(base_delay)
            
            # Extra delay every 5 requests to be extra conservative
            if request_count % 5 == 0:
                await asyncio.sleep(5)
        
        logger.info(f"Collected {len(data_points)} data points for {series_id}")
        return data_points
    
    async def _fetch_interest_rate_data(
        self, 
        series_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch interest rate data - placeholder for future implementation."""
        # Interest rates might use different endpoints
        # For now, return empty list
        logger.info(f"Interest rate data collection for {series_id} not yet implemented")
        return []
    
    def _parse_observation(self, observation: Dict[str, Any], series_id: str) -> Optional[Dict[str, Any]]:
        """Parse a single observation from the REST API response."""
        try:
            # Expected JSON structure from the new API
            date_str = observation.get("date", "")
            value = observation.get("value", None)
            
            if not date_str or value is None:
                return None
            
            # Parse date
            date = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Convert value to float
            if isinstance(value, str):
                value = float(value)
            elif not isinstance(value, (int, float)):
                return None
            
            series_config = self.SERIES_CONFIG.get(series_id, RiksbankSeries("", "", "", ""))
            
            return {
                "series_id": series_id,
                "date": date,
                "value": value,
                "description": series_config.description,
                "unit": series_config.unit
            }
            
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Error parsing observation: {e}")
            return None
    
    async def collect_all_series(self, days_back: int = 7) -> Dict[str, List[Dict[str, Any]]]:
        """Collect data for all configured series."""
        # Reduce default to 7 days to respect rate limits
        start_date = datetime.now() - timedelta(days=days_back)
        end_date = datetime.now()
        
        results = {}
        
        for series_id in self.SERIES_CONFIG.keys():
            logger.info(f"Collecting Riksbank data for series: {series_id}")
            data = await self.fetch_series_data(series_id, start_date, end_date)
            results[series_id] = data
            
            # Longer delay between series to respect API limits
            await asyncio.sleep(2)
        
        return results
    
    async def save_to_database(self, data: Dict[str, List[Dict[str, Any]]]) -> int:
        """Save collected data to database."""
        total_saved = 0
        
        with get_db_session() as session:
            for series_id, data_points in data.items():
                for point in data_points:
                    # Check if data point already exists
                    existing = session.query(RiksbankData).filter(
                        RiksbankData.series_id == point["series_id"],
                        RiksbankData.date == point["date"]
                    ).first()
                    
                    if not existing:
                        db_record = RiksbankData(
                            series_id=point["series_id"],
                            date=point["date"],
                            value=point["value"],
                            description=point["description"],
                            unit=point["unit"]
                        )
                        session.add(db_record)
                        total_saved += 1
        
        logger.info(f"Saved {total_saved} new Riksbank data points")
        return total_saved

# Example usage and testing
async def test_riksbank_collector():
    """Test the Riksbank collector."""
    async with RiksbankCollector() as collector:
        # Test fetching repo rate data
        data = await collector.fetch_series_data("SEKREPORATE")
        print(f"Fetched {len(data)} data points for repo rate")
        
        if data:
            latest = data[-1]
            print(f"Latest repo rate: {latest['value']}% on {latest['date']}")

if __name__ == "__main__":
    # Initialize database first
    from backend.database import init_database
    init_database()
    
    # Run test
    asyncio.run(test_riksbank_collector())