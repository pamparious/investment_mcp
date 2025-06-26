"""
Unified Swedish economic data collector for Investment MCP System.

This module consolidates Riksbanken and SCB data collection into a single,
efficient collector for Swedish economic indicators.
"""

import aiohttp
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from sqlalchemy.orm import Session
from ..core.config import get_settings
from ..core.database import EconomicData, SwedishEconomicIndicator

logger = logging.getLogger(__name__)


class SwedishDataCollector:
    """Unified collector for Swedish economic data from Riksbanken and SCB."""
    
    # Key Riksbanken series for investment analysis
    RIKSBANK_SERIES = {
        "repo_rate": {
            "series_id": "SEKREPOPOL",
            "description": "Repo rate (policy rate)",
            "unit": "percent",
            "frequency": "daily",
            "impact_rating": "high"
        },
        "inflation_target": {
            "series_id": "KPIF",
            "description": "CPIF inflation rate",
            "unit": "percent",
            "frequency": "monthly", 
            "impact_rating": "high"
        },
        "sek_eur": {
            "series_id": "SEKEUR", 
            "description": "SEK/EUR exchange rate",
            "unit": "rate",
            "frequency": "daily",
            "impact_rating": "medium"
        },
        "sek_usd": {
            "series_id": "SEKUSD",
            "description": "SEK/USD exchange rate", 
            "unit": "rate",
            "frequency": "daily",
            "impact_rating": "medium"
        },
        "government_bond_10y": {
            "series_id": "SEGOV10Y",
            "description": "10-year government bond yield",
            "unit": "percent",
            "frequency": "daily",
            "impact_rating": "high"
        }
    }
    
    # Key SCB indicators for housing and demographic analysis
    SCB_INDICATORS = {
        "housing_prices": {
            "table_id": "BO0501",
            "description": "Housing price index",
            "unit": "index",
            "frequency": "quarterly",
            "impact_rating": "high"
        },
        "population": {
            "table_id": "BE0101",
            "description": "Population by region",
            "unit": "persons",
            "frequency": "monthly",
            "impact_rating": "low"
        },
        "unemployment": {
            "table_id": "AM0210",
            "description": "Unemployment rate by region",
            "unit": "percent", 
            "frequency": "monthly",
            "impact_rating": "medium"
        },
        "gdp": {
            "table_id": "NR0103",
            "description": "GDP by quarter",
            "unit": "million_sek",
            "frequency": "quarterly",
            "impact_rating": "high"
        }
    }
    
    def __init__(self):
        self.settings = get_settings()
        self.riksbank_base_url = self.settings.RIKSBANK_API_BASE
        self.scb_base_url = self.settings.SCB_API_BASE
        
    async def collect_all_economic_data(self, years: int = None) -> Dict[str, pd.DataFrame]:
        """Collect all Swedish economic data from both Riksbanken and SCB."""
        
        if years is None:
            years = min(self.settings.HISTORICAL_DATA_YEARS, 10)  # Limit to 10 years for economic data
        
        logger.info(f"Collecting {years} years of Swedish economic data")
        
        # Collect data from both sources concurrently
        riksbank_task = self.collect_riksbank_data(years)
        scb_task = self.collect_scb_data(years)
        
        riksbank_data, scb_data = await asyncio.gather(
            riksbank_task, scb_task, return_exceptions=True
        )
        
        results = {}
        
        # Process Riksbanken results
        if isinstance(riksbank_data, Exception):
            logger.error(f"Riksbanken data collection failed: {riksbank_data}")
        else:
            results.update(riksbank_data)
        
        # Process SCB results
        if isinstance(scb_data, Exception):
            logger.error(f"SCB data collection failed: {scb_data}")
        else:
            results.update(scb_data)
        
        return results
    
    async def collect_riksbank_data(self, years: int = 5) -> Dict[str, pd.DataFrame]:
        """Collect data from Riksbanken API."""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        results = {}
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for indicator_name, config in self.RIKSBANK_SERIES.items():
                task = self._fetch_riksbank_series(
                    session, indicator_name, config, start_date, end_date
                )
                tasks.append(task)
            
            series_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for indicator_name, result in zip(self.RIKSBANK_SERIES.keys(), series_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to collect Riksbanken {indicator_name}: {result}")
                    results[f"riksbank_{indicator_name}"] = pd.DataFrame()
                else:
                    results[f"riksbank_{indicator_name}"] = result
        
        return results
    
    async def _fetch_riksbank_series(
        self, 
        session: aiohttp.ClientSession,
        indicator_name: str,
        config: Dict[str, str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch a single series from Riksbanken API."""
        
        # Construct Riksbanken API URL
        series_id = config["series_id"]
        url = f"{self.riksbank_base_url}/swea/v1/crosssection/{series_id}"
        
        params = {
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "format": "json"
        }
        
        try:
            async with session.get(url, params=params, timeout=30) as response:
                if response.status != 200:
                    logger.warning(f"Riksbanken API returned status {response.status} for {series_id}")
                    return pd.DataFrame()
                
                data = await response.json()
                
                # Parse Riksbanken response format
                if not data or "value" not in data:
                    logger.warning(f"No data returned for Riksbanken series {series_id}")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                records = []
                for item in data["value"]:
                    if item.get("date") and item.get("value") is not None:
                        records.append({
                            "date": pd.to_datetime(item["date"]).date(),
                            "value": float(item["value"]),
                            "series_id": series_id,
                            "description": config["description"],
                            "unit": config["unit"],
                            "frequency": config["frequency"],
                            "source": "riksbank"
                        })
                
                df = pd.DataFrame(records)
                
                if not df.empty:
                    df = df.sort_values("date")
                    logger.info(f"Collected {len(df)} Riksbanken records for {indicator_name}")
                
                return df
                
        except Exception as e:
            logger.error(f"Error fetching Riksbanken series {series_id}: {e}")
            return pd.DataFrame()
    
    async def collect_scb_data(self, years: int = 5) -> Dict[str, pd.DataFrame]:
        """Collect data from SCB (Statistics Sweden) API."""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        results = {}
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for indicator_name, config in self.SCB_INDICATORS.items():
                task = self._fetch_scb_indicator(
                    session, indicator_name, config, start_date, end_date
                )
                tasks.append(task)
            
            indicator_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for indicator_name, result in zip(self.SCB_INDICATORS.keys(), indicator_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to collect SCB {indicator_name}: {result}")
                    results[f"scb_{indicator_name}"] = pd.DataFrame()
                else:
                    results[f"scb_{indicator_name}"] = result
        
        return results
    
    async def _fetch_scb_indicator(
        self,
        session: aiohttp.ClientSession,
        indicator_name: str,
        config: Dict[str, str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch a single indicator from SCB API."""
        
        table_id = config["table_id"]
        
        # SCB API has different endpoints for different tables
        # This is a simplified implementation - real SCB API requires specific queries
        url = f"{self.scb_base_url}/v1/en/ssd/{table_id}"
        
        try:
            # Note: This is a simplified implementation
            # Real SCB API requires complex JSON queries for each table
            # For now, we'll return empty DataFrame and log the attempt
            
            logger.info(f"SCB API collection for {indicator_name} requires custom implementation")
            
            # Placeholder data structure for development
            return pd.DataFrame({
                "date": [],
                "value": [],
                "table_id": [],
                "region": [],
                "description": [],
                "unit": [],
                "frequency": [],
                "source": []
            })
            
        except Exception as e:
            logger.error(f"Error fetching SCB indicator {table_id}: {e}")
            return pd.DataFrame()
    
    def save_to_database(self, data_dict: Dict[str, pd.DataFrame], db_session: Session) -> int:
        """Save collected Swedish economic data to database."""
        
        total_records = 0
        
        for data_key, data in data_dict.items():
            if data.empty:
                continue
            
            source = "riksbank" if "riksbank" in data_key else "scb"
            
            for _, row in data.iterrows():
                # Save to general economic data table
                existing = db_session.query(EconomicData).filter(
                    EconomicData.source == source,
                    EconomicData.series_id == row.get('series_id', data_key),
                    EconomicData.date == row['date']
                ).first()
                
                if existing:
                    # Update existing record
                    existing.value = row['value']
                    existing.description = row.get('description')
                    existing.unit = row.get('unit')
                    existing.frequency = row.get('frequency')
                    existing.updated_at = datetime.utcnow()
                else:
                    # Create new record
                    economic_data = EconomicData(
                        source=source,
                        series_id=row.get('series_id', data_key),
                        table_id=row.get('table_id'),
                        region=row.get('region', 'SE'),  # Default to Sweden
                        date=row['date'],
                        value=row['value'],
                        description=row.get('description'),
                        unit=row.get('unit'),
                        frequency=row.get('frequency')
                    )
                    db_session.add(economic_data)
                    total_records += 1
                
                # Also save to specialized Swedish indicators table if from known series
                if source == "riksbank" and row.get('series_id') in [s['series_id'] for s in self.RIKSBANK_SERIES.values()]:
                    # Find the indicator config
                    indicator_config = None
                    indicator_name = None
                    for name, config in self.RIKSBANK_SERIES.items():
                        if config['series_id'] == row.get('series_id'):
                            indicator_config = config
                            indicator_name = name
                            break
                    
                    if indicator_config:
                        existing_indicator = db_session.query(SwedishEconomicIndicator).filter(
                            SwedishEconomicIndicator.indicator_name == indicator_name,
                            SwedishEconomicIndicator.date == row['date']
                        ).first()
                        
                        if not existing_indicator:
                            # Calculate change from previous value
                            previous_record = db_session.query(SwedishEconomicIndicator).filter(
                                SwedishEconomicIndicator.indicator_name == indicator_name,
                                SwedishEconomicIndicator.date < row['date']
                            ).order_by(SwedishEconomicIndicator.date.desc()).first()
                            
                            previous_value = previous_record.value if previous_record else None
                            change_pct = None
                            if previous_value and previous_value != 0:
                                change_pct = ((row['value'] - previous_value) / previous_value) * 100
                            
                            swedish_indicator = SwedishEconomicIndicator(
                                indicator_name=indicator_name,
                                indicator_type=self._get_indicator_type(indicator_name),
                                date=row['date'],
                                value=row['value'],
                                previous_value=previous_value,
                                change_pct=change_pct,
                                unit=row.get('unit'),
                                frequency=row.get('frequency'),
                                source=source,
                                impact_rating=indicator_config.get('impact_rating', 'medium')
                            )
                            db_session.add(swedish_indicator)
        
        db_session.commit()
        logger.info(f"Saved {total_records} new economic data records to database")
        
        return total_records
    
    def _get_indicator_type(self, indicator_name: str) -> str:
        """Map indicator name to type category."""
        
        type_mapping = {
            "repo_rate": "interest_rate",
            "inflation_target": "inflation", 
            "sek_eur": "exchange_rate",
            "sek_usd": "exchange_rate",
            "government_bond_10y": "interest_rate",
            "housing_prices": "housing",
            "population": "demographic",
            "unemployment": "employment",
            "gdp": "gdp"
        }
        
        return type_mapping.get(indicator_name, "other")
    
    def get_latest_indicators(self, db_session: Session) -> Dict[str, Dict[str, Any]]:
        """Get the latest values for all Swedish economic indicators."""
        
        indicators = {}
        
        # Get latest value for each indicator
        for indicator_name in self.RIKSBANK_SERIES.keys():
            latest = db_session.query(SwedishEconomicIndicator).filter(
                SwedishEconomicIndicator.indicator_name == indicator_name
            ).order_by(SwedishEconomicIndicator.date.desc()).first()
            
            if latest:
                indicators[indicator_name] = {
                    "current_value": latest.value,
                    "previous_value": latest.previous_value,
                    "change_pct": latest.change_pct,
                    "date": latest.date.isoformat(),
                    "unit": latest.unit,
                    "impact_rating": latest.impact_rating,
                    "description": self.RIKSBANK_SERIES[indicator_name]["description"]
                }
        
        return indicators
    
    async def collect_recent_data(self, days: int = 30) -> Dict[str, pd.DataFrame]:
        """Collect recent Swedish economic data for daily updates."""
        
        logger.info(f"Collecting recent {days} days of Swedish economic data")
        
        # For economic data, we typically collect less frequent updates
        # Use shorter time window but same collection method
        years = max(1, days / 365)  # Convert days to fraction of year
        
        return await self.collect_all_economic_data(years=years)