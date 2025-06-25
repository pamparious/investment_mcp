"""Configuration settings for the Investment MCP System."""

import os
from functools import lru_cache
from typing import Dict, List, Optional

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Application settings using Pydantic BaseSettings."""
    
    # Application
    app_name: str = Field(default="Investment MCP System", env="APP_NAME")
    version: str = Field(default="4.0.0", env="VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Database
    database_url: str = Field(default="sqlite:///./investment_data.db", env="DATABASE_URL")
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")
    database_pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")
    
    # Cache
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    
    # Security
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    allowed_origins: List[str] = Field(default=["*"], env="ALLOWED_ORIGINS")
    
    # External APIs
    riksbank_base_url: str = Field(
        default="https://api.riksbank.se", env="RIKSBANK_BASE_URL"
    )
    scb_base_url: str = Field(
        default="https://api.scb.se", env="SCB_BASE_URL"
    )
    
    # AI Configuration
    ai_provider: str = Field(default="openai", env="AI_PROVIDER")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    ai_model: str = Field(default="gpt-3.5-turbo", env="AI_MODEL")
    ai_timeout: int = Field(default=60, env="AI_TIMEOUT")
    
    # Data Collection
    data_collection_interval: int = Field(default=3600, env="DATA_COLLECTION_INTERVAL")
    historical_years: int = Field(default=20, env="HISTORICAL_YEARS")
    data_quality_threshold: float = Field(default=0.8, env="DATA_QUALITY_THRESHOLD")
    
    # Analysis
    portfolio_optimization_method: str = Field(
        default="mean_variance", env="PORTFOLIO_OPTIMIZATION_METHOD"
    )
    risk_free_rate: float = Field(default=0.02, env="RISK_FREE_RATE")
    monte_carlo_simulations: int = Field(default=10000, env="MONTE_CARLO_SIMULATIONS")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment value."""
        if v not in ["development", "staging", "production"]:
            raise ValueError("Environment must be development, staging, or production")
        return v
    
    @validator("ai_provider")
    def validate_ai_provider(cls, v):
        """Validate AI provider."""
        if v not in ["openai", "anthropic", "azure"]:
            raise ValueError("AI provider must be openai, anthropic, or azure")
        return v
    
    @validator("portfolio_optimization_method")
    def validate_optimization_method(cls, v):
        """Validate portfolio optimization method."""
        valid_methods = ["mean_variance", "risk_parity", "black_litterman", "genetic"]
        if v not in valid_methods:
            raise ValueError(f"Optimization method must be one of {valid_methods}")
        return v
    
    @property
    def is_development(self) -> bool:
        """Check if environment is development."""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if environment is production."""
        return self.environment == "production"
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Fund Universe Constants
FUND_UNIVERSE = {
    "global_equity": ["DNB Global Indeks S"],
    "emerging_markets": ["Avanza Emerging Markets"],
    "europe": ["Storebrand Europa A SEK"],
    "nordic": ["DNB Norden Indeks S"],
    "sweden": ["PLUS Allabolag Sverige Index"],
    "usa": ["Avanza USA"],
    "japan": ["Storebrand Japan A SEK"],
    "small_cap": ["Handelsbanken Global småb index"],
    "commodities": ["Xetra-gold ETC"],
    "crypto": ["Virtune bitcoin prime etp", "XBT Ether one"],
    "real_estate": ["Plus fastigheter Sverige index"]
}

# Flatten fund universe for easy access
ALL_FUNDS = [fund for funds in FUND_UNIVERSE.values() for fund in funds]

# Fund to ticker mapping for data collection
FUND_TICKER_MAPPING = {
    "DNB Global Indeks S": "DNB0A.ST",
    "Avanza Emerging Markets": "AVZAEM.ST", 
    "Storebrand Europa A SEK": "STOEUR.ST",
    "DNB Norden Indeks S": "DNB0N.ST",
    "PLUS Allabolag Sverige Index": "PLUSAB.ST",
    "Avanza USA": "AVZAUS.ST",
    "Storebrand Japan A SEK": "STOJAP.ST",
    "Handelsbanken Global småb index": "HASMAB.ST",
    "Xetra-gold ETC": "4GLD.DE",
    "Virtune bitcoin prime etp": "BITCOIN.ST",
    "XBT Ether one": "ETHER.ST",
    "Plus fastigheter Sverige index": "PLUSFA.ST"
}

# Risk profiles
RISK_PROFILES = {
    "conservative": {
        "expected_return": 0.05,
        "max_volatility": 0.12,
        "max_equity_allocation": 0.60,
        "description": "Capital preservation focused"
    },
    "balanced": {
        "expected_return": 0.07,
        "max_volatility": 0.16,
        "max_equity_allocation": 0.80,
        "description": "Balanced growth and protection"
    },
    "aggressive": {
        "expected_return": 0.10,
        "max_volatility": 0.25,
        "max_equity_allocation": 1.00,
        "description": "Growth focused"
    }
}


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_fund_ticker(fund_name: str) -> Optional[str]:
    """Get ticker symbol for fund name."""
    return FUND_TICKER_MAPPING.get(fund_name)


def get_fund_category(fund_name: str) -> Optional[str]:
    """Get category for fund name."""
    for category, funds in FUND_UNIVERSE.items():
        if fund_name in funds:
            return category
    return None


def validate_fund_name(fund_name: str) -> bool:
    """Validate if fund name is in our universe."""
    return fund_name in ALL_FUNDS


def get_risk_profile_config(profile: str) -> Optional[Dict]:
    """Get risk profile configuration."""
    return RISK_PROFILES.get(profile)