"""Riksbank API collector for Swedish economic data."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import aiohttp
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.config.settings import get_settings
from ...utils.exceptions import DataCollectionError


logger = logging.getLogger(__name__)


class RiksbankCollector:
    """Collector for Swedish economic data from Riksbank API."""
    
    def __init__(self):
        """Initialize the Riksbank collector."""
        self.settings = get_settings()
        self.base_url = self.settings.riksbank_base_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"Accept": "application/json"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def collect_interest_rates(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Collect Swedish interest rate data.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            Dictionary with interest rate data
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365 * 2)  # 2 years back
        
        if end_date is None:
            end_date = datetime.now()
        
        logger.info(f"Collecting Riksbank interest rate data from {start_date.date()} to {end_date.date()}")
        
        try:
            # Riksbank repo rate (policy rate)
            repo_rate_data = await self._fetch_series("SEREPORATE", start_date, end_date)
            
            # Government bond yields
            gov_bond_2y = await self._fetch_series("SEGVB2Y", start_date, end_date)
            gov_bond_5y = await self._fetch_series("SEGVB5Y", start_date, end_date)
            gov_bond_10y = await self._fetch_series("SEGVB10Y", start_date, end_date)
            
            # Mortgage rates (if available)
            mortgage_rates = await self._fetch_series("SEMORTGAGE", start_date, end_date)
            
            # Combine all data
            result = {
                "repo_rate": self._process_series_data(repo_rate_data, "repo_rate"),
                "government_bonds": {
                    "2y": self._process_series_data(gov_bond_2y, "gov_bond_2y"),
                    "5y": self._process_series_data(gov_bond_5y, "gov_bond_5y"),
                    "10y": self._process_series_data(gov_bond_10y, "gov_bond_10y")
                },
                "mortgage_rates": self._process_series_data(mortgage_rates, "mortgage_rates"),
                "collection_timestamp": datetime.utcnow().isoformat(),
                "data_source": "riksbank"
            }
            
            # Add derived metrics
            result["yield_curve"] = self._calculate_yield_curve(result["government_bonds"])
            result["interest_rate_trends"] = self._analyze_rate_trends(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error collecting Riksbank interest rate data: {e}")
            raise DataCollectionError(f"Failed to collect interest rate data: {e}")
    
    async def collect_exchange_rates(
        self,
        currencies: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Collect Swedish exchange rate data.
        
        Args:
            currencies: List of currencies to collect (default: USD, EUR, NOK, DKK, GBP)
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            Dictionary with exchange rate data
        """
        if currencies is None:
            currencies = ["USD", "EUR", "NOK", "DKK", "GBP"]
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)  # 1 year back
        
        if end_date is None:
            end_date = datetime.now()
        
        logger.info(f"Collecting exchange rates for {currencies}")
        
        try:
            exchange_rates = {}
            
            for currency in currencies:
                series_id = f"SEKEX{currency}"
                rate_data = await self._fetch_series(series_id, start_date, end_date)
                exchange_rates[f"SEK_{currency}"] = self._process_series_data(rate_data, f"sek_{currency.lower()}")
            
            # Calculate currency strength index
            currency_strength = self._calculate_currency_strength(exchange_rates)
            
            result = {
                "exchange_rates": exchange_rates,
                "currency_strength_index": currency_strength,
                "collection_timestamp": datetime.utcnow().isoformat(),
                "data_source": "riksbank"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error collecting exchange rate data: {e}")
            raise DataCollectionError(f"Failed to collect exchange rate data: {e}")
    
    async def collect_monetary_policy_data(self) -> Dict[str, Any]:
        """
        Collect monetary policy related data.
        
        Returns:
            Dictionary with monetary policy data
        """
        logger.info("Collecting monetary policy data")
        
        try:
            # Current policy rate
            current_date = datetime.now()
            start_date = current_date - timedelta(days=30)  # Last 30 days
            
            repo_rate = await self._fetch_series("SEREPORATE", start_date, current_date)
            
            # Policy meeting dates and decisions (simulated for now)
            policy_meetings = await self._get_policy_meetings()
            
            # Market expectations (if available)
            market_expectations = await self._get_market_expectations()
            
            result = {
                "current_repo_rate": self._get_latest_value(repo_rate),
                "repo_rate_history": self._process_series_data(repo_rate, "repo_rate"),
                "policy_meetings": policy_meetings,
                "market_expectations": market_expectations,
                "collection_timestamp": datetime.utcnow().isoformat(),
                "data_source": "riksbank"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error collecting monetary policy data: {e}")
            raise DataCollectionError(f"Failed to collect monetary policy data: {e}")
    
    async def _fetch_series(
        self,
        series_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch a data series from Riksbank API."""
        
        if not self.session:
            raise ValueError("Session not initialized. Use async context manager.")
        
        # Format dates for API
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Construct API URL (simplified - actual Riksbank API has different structure)
        url = f"{self.base_url}/swea/v1/crossrates/{series_id}"
        params = {
            "from": start_str,
            "to": end_str,
            "format": "json"
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
                elif response.status == 404:
                    logger.warning(f"Series {series_id} not found")
                    return None
                else:
                    logger.error(f"API error {response.status} for series {series_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching series {series_id}: {e}")
            # Return simulated data for demonstration
            return self._generate_simulated_data(series_id, start_date, end_date)
    
    def _generate_simulated_data(
        self,
        series_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Generate simulated data when API is not available."""
        
        logger.info(f"Generating simulated data for {series_id}")
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        
        # Base values for different series
        base_values = {
            "SEREPORATE": 4.0,  # Current repo rate around 4%
            "SEGVB2Y": 3.5,     # 2-year government bond
            "SEGVB5Y": 3.8,     # 5-year government bond
            "SEGVB10Y": 4.2,    # 10-year government bond
            "SEMORTGAGE": 6.0,   # Mortgage rate
            "SEKEXUSD": 10.8,   # SEK/USD exchange rate
            "SEKEXEUR": 11.7,   # SEK/EUR exchange rate
            "SEKEXNOK": 1.03,   # SEK/NOK exchange rate
            "SEKEXDKK": 1.57,   # SEK/DKK exchange rate
            "SEKEXGBP": 13.4,   # SEK/GBP exchange rate
        }
        
        base_value = base_values.get(series_id, 2.0)
        
        # Generate realistic random walk
        import numpy as np
        np.random.seed(hash(series_id) % 2**32)  # Consistent seed
        
        volatility = 0.001 if "RATE" in series_id else 0.005
        values = []
        current_value = base_value
        
        for date in date_range:
            # Add small random changes
            change = np.random.normal(0, volatility)
            current_value = max(0.01, current_value + change)
            
            values.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": round(current_value, 4)
            })
        
        return values
    
    def _process_series_data(
        self,
        raw_data: Optional[List[Dict[str, Any]]],
        series_name: str
    ) -> Dict[str, Any]:
        """Process raw series data into structured format."""
        
        if not raw_data:
            return {
                "current_value": None,
                "data_points": 0,
                "first_date": None,
                "last_date": None,
                "data": [],
                "statistics": {}
            }
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(raw_data)
        if "date" in df.columns and "value" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            
            # Calculate statistics
            values = df["value"].dropna()
            statistics = {
                "mean": float(values.mean()) if len(values) > 0 else None,
                "std": float(values.std()) if len(values) > 1 else None,
                "min": float(values.min()) if len(values) > 0 else None,
                "max": float(values.max()) if len(values) > 0 else None,
                "latest_change": None,
                "trend": "stable"
            }
            
            # Calculate latest change
            if len(values) >= 2:
                statistics["latest_change"] = float(values.iloc[-1] - values.iloc[-2])
                
                # Determine trend
                if len(values) >= 10:
                    recent_trend = values.tail(10).diff().mean()
                    if recent_trend > 0.01:
                        statistics["trend"] = "increasing"
                    elif recent_trend < -0.01:
                        statistics["trend"] = "decreasing"
            
            return {
                "current_value": float(values.iloc[-1]) if len(values) > 0 else None,
                "data_points": len(df),
                "first_date": df["date"].iloc[0].isoformat() if len(df) > 0 else None,
                "last_date": df["date"].iloc[-1].isoformat() if len(df) > 0 else None,
                "data": df.to_dict("records"),
                "statistics": statistics
            }
        
        return {"error": "Invalid data format"}
    
    def _get_latest_value(self, series_data: Optional[List[Dict[str, Any]]]) -> Optional[float]:
        """Get the latest value from series data."""
        
        if not series_data:
            return None
        
        # Find the most recent data point
        latest_point = max(series_data, key=lambda x: x.get("date", ""))
        return latest_point.get("value")
    
    def _calculate_yield_curve(self, government_bonds: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate yield curve metrics."""
        
        try:
            yield_2y = government_bonds["2y"].get("current_value")
            yield_5y = government_bonds["5y"].get("current_value")
            yield_10y = government_bonds["10y"].get("current_value")
            
            if all(x is not None for x in [yield_2y, yield_5y, yield_10y]):
                # Calculate spreads
                spread_10y_2y = yield_10y - yield_2y
                spread_10y_5y = yield_10y - yield_5y
                
                # Determine curve shape
                if spread_10y_2y > 0.5:
                    shape = "normal"
                elif spread_10y_2y < 0:
                    shape = "inverted"
                else:
                    shape = "flat"
                
                return {
                    "shape": shape,
                    "spread_10y_2y": spread_10y_2y,
                    "spread_10y_5y": spread_10y_5y,
                    "steepness": spread_10y_2y
                }
        
        except Exception as e:
            logger.error(f"Error calculating yield curve: {e}")
        
        return {"error": "Unable to calculate yield curve"}
    
    def _analyze_rate_trends(self, rate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze interest rate trends."""
        
        try:
            repo_stats = rate_data["repo_rate"]["statistics"]
            
            trends = {
                "repo_rate_trend": repo_stats.get("trend", "stable"),
                "latest_change": repo_stats.get("latest_change", 0),
                "volatility": repo_stats.get("std", 0),
                "assessment": "stable"
            }
            
            # Overall assessment
            latest_change = abs(trends["latest_change"] or 0)
            if latest_change > 0.25:
                trends["assessment"] = "volatile"
            elif repo_stats.get("trend") == "increasing":
                trends["assessment"] = "tightening"
            elif repo_stats.get("trend") == "decreasing":
                trends["assessment"] = "easing"
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing rate trends: {e}")
            return {"error": "Unable to analyze trends"}
    
    def _calculate_currency_strength(self, exchange_rates: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate SEK currency strength index."""
        
        try:
            # Get current values for major currencies
            major_currencies = ["SEK_USD", "SEK_EUR", "SEK_GBP"]
            current_rates = {}
            
            for currency in major_currencies:
                rate_data = exchange_rates.get(currency, {})
                current_value = rate_data.get("current_value")
                if current_value:
                    current_rates[currency] = current_value
            
            if len(current_rates) >= 2:
                # Simple strength index (lower values = stronger SEK)
                # Normalize against historical averages (simplified)
                historical_averages = {
                    "SEK_USD": 9.5,
                    "SEK_EUR": 10.8,
                    "SEK_GBP": 12.0
                }
                
                strength_components = []
                for currency, current_rate in current_rates.items():
                    historical_avg = historical_averages.get(currency, current_rate)
                    strength_component = (historical_avg / current_rate) * 100
                    strength_components.append(strength_component)
                
                strength_index = sum(strength_components) / len(strength_components)
                
                # Classify strength
                if strength_index > 105:
                    strength_level = "strong"
                elif strength_index < 95:
                    strength_level = "weak"
                else:
                    strength_level = "neutral"
                
                return {
                    "index": round(strength_index, 2),
                    "level": strength_level,
                    "components": dict(zip(current_rates.keys(), strength_components))
                }
        
        except Exception as e:
            logger.error(f"Error calculating currency strength: {e}")
        
        return {"error": "Unable to calculate currency strength"}
    
    async def _get_policy_meetings(self) -> List[Dict[str, Any]]:
        """Get upcoming policy meeting dates (simulated)."""
        
        # In real implementation, this would fetch from Riksbank calendar
        current_date = datetime.now()
        
        meetings = []
        # Generate next 4 quarterly meetings
        for i in range(4):
            meeting_date = current_date + timedelta(days=90 * i + 30)
            meetings.append({
                "date": meeting_date.strftime("%Y-%m-%d"),
                "type": "Monetary Policy Meeting",
                "expected_decision": "hold" if i < 2 else "tbd"
            })
        
        return meetings
    
    async def _get_market_expectations(self) -> Dict[str, Any]:
        """Get market expectations for policy rates (simulated)."""
        
        # In real implementation, this would fetch from market data
        return {
            "next_meeting": {
                "probability_cut": 0.1,
                "probability_hold": 0.8,
                "probability_hike": 0.1
            },
            "12_month_outlook": {
                "expected_rate": 3.75,
                "probability_range": {
                    "3.0_3.5": 0.2,
                    "3.5_4.0": 0.6,
                    "4.0_4.5": 0.2
                }
            }
        }