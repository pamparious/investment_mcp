"""Enhanced collector for comprehensive Swedish economic data."""

import aiohttp
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import json
import numpy as np

logger = logging.getLogger(__name__)


class SwedishEconomicCollector:
    """Enhanced collector for comprehensive Swedish economic data."""
    
    def __init__(self):
        """Initialize the Swedish economic data collector."""
        self.riksbank_base = "https://api.riksbank.se/swea/v1/crossrates"
        self.scb_base = "https://api.scb.se/OV0104/v1/doris"
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"Content-Type": "application/json"}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            
    async def get_comprehensive_economic_data(self, years_back: int = 5) -> Dict[str, Any]:
        """Collect comprehensive Swedish economic indicators."""
        
        logger.info(f"Collecting comprehensive Swedish economic data for {years_back} years")
        
        async with self:
            tasks = [
                self.get_interest_rate_trends(),
                self.get_currency_trends(), 
                self.get_inflation_data(),
                self.get_housing_market_data(),
                self.get_economic_growth_indicators(),
                self.get_employment_data(),
                self.get_manufacturing_pmi(),
                self.get_consumer_confidence()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return {
                "interest_rates": results[0] if not isinstance(results[0], Exception) else {"error": str(results[0])},
                "currency": results[1] if not isinstance(results[1], Exception) else {"error": str(results[1])},
                "inflation": results[2] if not isinstance(results[2], Exception) else {"error": str(results[2])},
                "housing": results[3] if not isinstance(results[3], Exception) else {"error": str(results[3])},
                "gdp_growth": results[4] if not isinstance(results[4], Exception) else {"error": str(results[4])},
                "employment": results[5] if not isinstance(results[5], Exception) else {"error": str(results[5])},
                "manufacturing": results[6] if not isinstance(results[6], Exception) else {"error": str(results[6])},
                "consumer_confidence": results[7] if not isinstance(results[7], Exception) else {"error": str(results[7])},
                "collection_timestamp": datetime.utcnow().isoformat(),
                "economic_cycle_phase": await self.analyze_economic_cycle_phase()
            }
    
    async def get_interest_rate_trends(self) -> Dict[str, Any]:
        """Get detailed interest rate trends and predictions."""
        try:
            # Simulated Riksbank interest rate data
            # In production, this would connect to actual Riksbank API
            current_repo_rate = 4.0  # Current Swedish repo rate
            
            # Generate trend data (last 2 years)
            dates = pd.date_range(start=datetime.now() - timedelta(days=730), 
                                end=datetime.now(), freq='M')
            
            # Simulate rate changes
            base_rate = 0.0
            rates = []
            for i, date in enumerate(dates):
                if i < 12:  # First year - low rates
                    rate = base_rate + (i * 0.1)
                else:  # Second year - rate increases
                    rate = min(4.0, base_rate + 1.2 + ((i-12) * 0.25))
                rates.append(rate)
            
            trend_data = pd.DataFrame({
                'date': dates,
                'repo_rate': rates,
                'government_bond_10y': [r + 1.5 + np.random.normal(0, 0.2) for r in rates],
                'mortgage_rate': [r + 2.0 + np.random.normal(0, 0.3) for r in rates]
            })
            
            return {
                "current_repo_rate": current_repo_rate,
                "rate_trend": "increasing",
                "rate_change_12m": 3.5,  # Increased 3.5% over last 12 months
                "next_decision_date": (datetime.now() + timedelta(days=45)).isoformat(),
                "expected_direction": "stable_to_slight_increase",
                "historical_data": trend_data.to_dict('records')[-12:],  # Last 12 months
                "real_rate": current_repo_rate - 2.1,  # Minus inflation
                "rate_cycle_phase": "late_tightening"
            }
            
        except Exception as e:
            logger.error(f"Error collecting interest rate data: {e}")
            return {"error": str(e)}
    
    async def get_currency_trends(self) -> Dict[str, Any]:
        """Get SEK exchange rate trends vs major currencies."""
        try:
            # Simulated currency data
            # In production, connect to Riksbank exchange rate API
            
            current_rates = {
                "SEK_USD": 10.85,
                "SEK_EUR": 11.72,
                "SEK_GBP": 13.45,
                "SEK_NOK": 1.03,
                "SEK_DKK": 1.57
            }
            
            # Generate volatility and trend data
            return {
                "current_rates": current_rates,
                "sek_strength_index": 45.2,  # 0-100, lower = weaker SEK
                "volatility_30d": {
                    "SEK_USD": 8.5,  # % annualized
                    "SEK_EUR": 6.2,
                    "SEK_GBP": 11.3
                },
                "trend_6m": {
                    "SEK_USD": "weakening",  # SEK getting weaker vs USD
                    "SEK_EUR": "stable",
                    "SEK_GBP": "strengthening"
                },
                "purchasing_power_parity": {
                    "SEK_USD": 9.80,  # Fair value estimate
                    "SEK_EUR": 11.50
                },
                "carry_trade_attractiveness": 6.5,  # 1-10 scale
                "commodity_correlation": 0.72  # SEK correlation with commodity prices
            }
            
        except Exception as e:
            logger.error(f"Error collecting currency data: {e}")
            return {"error": str(e)}
    
    async def get_inflation_data(self) -> Dict[str, Any]:
        """Get comprehensive Swedish inflation data."""
        try:
            # Simulated SCB inflation data
            
            return {
                "current_cpi": 2.1,  # Current CPI inflation %
                "current_cpif": 1.8,  # CPI fixed interest rate
                "core_inflation": 2.3,  # Excluding energy and food
                "energy_inflation": -5.2,  # Energy price changes
                "food_inflation": 4.8,   # Food price changes
                "housing_inflation": 3.1, # Housing costs
                "services_inflation": 3.4,
                "inflation_target": 2.0,  # Riksbank target
                "target_deviation": 0.1,  # How far from target
                "inflation_expectations": {
                    "1_year": 2.2,
                    "2_years": 2.0,
                    "5_years": 2.1,
                    "long_term": 2.0
                },
                "breakeven_rates": {
                    "5_year": 2.15,
                    "10_year": 2.05
                },
                "trend_direction": "declining",
                "peak_inflation": 10.8,  # Peak during recent cycle
                "inflation_cycle_phase": "disinflation"
            }
            
        except Exception as e:
            logger.error(f"Error collecting inflation data: {e}")
            return {"error": str(e)}
    
    async def get_housing_market_data(self) -> Dict[str, Any]:
        """Get comprehensive Swedish housing market data."""
        try:
            # Simulated housing market data
            
            return {
                "house_price_index": 287.5,  # Index value
                "price_change_1m": -0.8,     # Monthly change %
                "price_change_3m": -2.4,     # Quarterly change %
                "price_change_12m": -12.5,   # Annual change %
                "regional_variation": {
                    "stockholm": -15.2,
                    "gothenburg": -10.8,
                    "malmo": -9.5,
                    "sweden_total": -12.5
                },
                "apartment_vs_houses": {
                    "apartments": -14.1,
                    "houses": -10.9
                },
                "housing_starts": {
                    "current_annual_rate": 42000,
                    "change_vs_previous_year": -22.5
                },
                "mortgage_volumes": {
                    "new_mortgages_billions_sek": 25.8,
                    "change_vs_previous_month": -8.2
                },
                "affordability_index": 78.5,  # 100 = normal
                "household_debt_to_income": 185.2,  # %
                "market_sentiment": "weak",
                "cycle_phase": "correction"
            }
            
        except Exception as e:
            logger.error(f"Error collecting housing data: {e}")
            return {"error": str(e)}
    
    async def get_economic_growth_indicators(self) -> Dict[str, Any]:
        """Get Swedish economic growth indicators."""
        try:
            # Simulated GDP and growth data
            
            return {
                "gdp_growth_quarterly": 0.2,    # Quarter-over-quarter %
                "gdp_growth_annual": 1.1,       # Year-over-year %
                "gdp_trend_growth": 1.8,        # Long-term trend
                "productivity_growth": 0.5,      # Annual productivity growth
                "business_investment_growth": -2.3,
                "consumer_spending_growth": 1.8,
                "export_growth": 2.4,
                "import_growth": 1.9,
                "industrial_production": {
                    "monthly_change": 0.5,
                    "annual_change": -1.2,
                    "trend": "stabilizing"
                },
                "services_pmi": 52.3,            # >50 = expansion
                "composite_pmi": 51.8,
                "economic_sentiment": 98.2,      # EU index
                "leading_indicators": {
                    "cli": 99.8,                 # Composite Leading Indicator
                    "trend": "neutral"
                },
                "recession_probability": 15.2,   # % probability next 12m
                "growth_cycle_phase": "slow_growth"
            }
            
        except Exception as e:
            logger.error(f"Error collecting growth data: {e}")
            return {"error": str(e)}
    
    async def get_employment_data(self) -> Dict[str, Any]:
        """Get Swedish employment data."""
        try:
            # Simulated employment data
            
            return {
                "unemployment_rate": 7.8,        # %
                "youth_unemployment": 23.5,      # 15-24 years
                "long_term_unemployment": 1.9,   # % of workforce
                "employment_rate": 69.2,         # % of working age
                "participation_rate": 75.1,      # % of working age
                "job_vacancies": 98500,          # Number of open positions
                "job_vacancy_rate": 1.9,         # % of total jobs
                "employment_change_12m": 45000,   # Net job creation
                "sectoral_employment": {
                    "manufacturing": -2.1,       # % change
                    "services": 1.8,
                    "construction": -5.2,
                    "public_sector": 0.5
                },
                "wage_growth": 3.4,              # Annual wage growth %
                "real_wage_growth": 1.3,         # Adjusted for inflation
                "hours_worked": 1.2,             # % change
                "labor_market_tightness": 0.85,  # Vacancies/unemployed
                "employment_cycle_phase": "cooling"
            }
            
        except Exception as e:
            logger.error(f"Error collecting employment data: {e}")
            return {"error": str(e)}
    
    async def get_manufacturing_pmi(self) -> Dict[str, Any]:
        """Get Swedish manufacturing PMI data."""
        try:
            # Simulated PMI data
            
            return {
                "manufacturing_pmi": 48.2,       # <50 = contraction
                "new_orders": 47.8,
                "production": 49.1,
                "employment": 46.5,
                "supplier_deliveries": 52.3,
                "inventories": 48.7,
                "prices_paid": 51.2,
                "export_orders": 45.9,
                "pmi_trend": "declining",
                "sector_breakdown": {
                    "automotive": 44.2,
                    "machinery": 49.8,
                    "chemicals": 51.1,
                    "forestry": 52.7,
                    "steel": 46.3
                },
                "regional_pmi": {
                    "south": 49.1,
                    "west": 47.8,
                    "north": 50.2
                },
                "diffusion_index": 42.3,         # % of positive responses
                "forward_looking_indicators": {
                    "future_production": 54.2,
                    "future_employment": 48.9,
                    "business_expectations": 52.1
                }
            }
            
        except Exception as e:
            logger.error(f"Error collecting PMI data: {e}")
            return {"error": str(e)}
    
    async def get_consumer_confidence(self) -> Dict[str, Any]:
        """Get Swedish consumer confidence data."""
        try:
            # Simulated consumer confidence data
            
            return {
                "consumer_confidence_index": -8.2,  # Net balance
                "personal_finances_now": -5.1,
                "personal_finances_12m": 3.8,
                "general_economy_now": -15.4,
                "general_economy_12m": -2.7,
                "unemployment_expectations": 25.3,  # % expecting increase
                "major_purchases": -18.9,           # Intentions
                "savings_propensity": 12.4,         # % planning to save more
                "house_price_expectations": -22.1,  # Net balance
                "inflation_expectations": 2.2,      # %
                "confidence_trend": "improving",
                "demographic_breakdown": {
                    "18_29": -12.4,
                    "30_49": -6.8,
                    "50_64": -7.1,
                    "65_plus": -4.2
                },
                "regional_confidence": {
                    "stockholm": -6.2,
                    "gothenburg": -8.9,
                    "malmo": -9.1,
                    "rest_of_sweden": -8.7
                }
            }
            
        except Exception as e:
            logger.error(f"Error collecting consumer confidence data: {e}")
            return {"error": str(e)}
    
    async def analyze_economic_cycle_phase(self) -> str:
        """Determine current Swedish economic cycle phase."""
        try:
            # This would analyze multiple indicators to determine cycle phase
            # For now, return a simulated analysis
            
            # Factors to consider:
            # - GDP growth trend
            # - Interest rates direction  
            # - Inflation vs target
            # - Employment trends
            # - PMI levels
            # - Consumer confidence
            
            # Based on current simulated indicators:
            # - GDP growth: 1.1% (below trend)
            # - Interest rates: 4.0% (restrictive)
            # - Inflation: 2.1% (near target, declining)
            # - PMI: 48.2 (contraction)
            # - Unemployment: 7.8% (elevated)
            
            return "late_cycle_slowdown"  # Other options: expansion, peak, contraction, trough, recovery
            
        except Exception as e:
            logger.error(f"Error analyzing economic cycle: {e}")
            return "uncertain"
    
    def analyze_interest_rate_trend(self, interest_data: Dict) -> float:
        """Analyze interest rate trend direction."""
        if not interest_data or "error" in interest_data:
            return 0.0
        return interest_data.get("rate_change_12m", 0.0)
    
    def analyze_inflation_trend(self, inflation_data: Dict) -> float:
        """Analyze inflation trend."""
        if not inflation_data or "error" in inflation_data:
            return 0.02  # Default 2%
        return inflation_data.get("current_cpi", 2.0) / 100.0
    
    def analyze_gdp_trend(self, gdp_data: Dict) -> float:
        """Analyze GDP growth trend."""
        if not gdp_data or "error" in gdp_data:
            return 0.0
        return gdp_data.get("gdp_growth_annual", 0.0) / 100.0
    
    def analyze_employment_trend(self, employment_data: Dict) -> float:
        """Analyze employment trend."""
        if not employment_data or "error" in employment_data:
            return 0.0
        # Convert unemployment rate to employment trend (lower unemployment = positive trend)
        unemployment_rate = employment_data.get("unemployment_rate", 7.0)
        return (8.0 - unemployment_rate) / 100.0  # Normalize around 8% baseline