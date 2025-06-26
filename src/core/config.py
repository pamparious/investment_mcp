"""
Unified configuration management for Investment MCP System.

This module consolidates all configuration from the various scattered config files
into a single, comprehensive configuration system using Pydantic BaseSettings.
"""

import os
from typing import List, Optional, Dict, Any
from functools import lru_cache

from pydantic import Field, validator
from pydantic_settings import BaseSettings


# Fund Universe - Exact 12 funds as specified
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

TRADEABLE_FUNDS = {
    "DNB_GLOBAL_INDEKS_S": {
        "name": "DNB Global Indeks S",
        "type": "global_equity",
        "description": "Global index fund tracking developed markets",
        "currency": "SEK",
        "risk_level": "medium",
        "expense_ratio": 0.15,
        "category": "global_equity"
    },
    "AVANZA_EMERGING_MARKETS": {
        "name": "Avanza Emerging Markets",
        "type": "emerging_markets",
        "description": "Emerging markets equity fund",
        "currency": "SEK", 
        "risk_level": "high",
        "expense_ratio": 0.50,
        "category": "emerging_markets"
    },
    "STOREBRAND_EUROPA_A_SEK": {
        "name": "Storebrand Europa A SEK",
        "type": "european_equity",
        "description": "European equity fund",
        "currency": "SEK",
        "risk_level": "medium",
        "expense_ratio": 0.75,
        "category": "european_equity"
    },
    "DNB_NORDEN_INDEKS_S": {
        "name": "DNB Norden Indeks S", 
        "type": "nordic_equity",
        "description": "Nordic index fund",
        "currency": "SEK",
        "risk_level": "medium",
        "expense_ratio": 0.20,
        "category": "nordic_equity"
    },
    "PLUS_ALLABOLAG_SVERIGE_INDEX": {
        "name": "PLUS Allabolag Sverige Index",
        "type": "swedish_equity", 
        "description": "Swedish equity index fund",
        "currency": "SEK",
        "risk_level": "medium",
        "expense_ratio": 0.10,
        "category": "swedish_equity"
    },
    "AVANZA_USA": {
        "name": "Avanza USA",
        "type": "us_equity",
        "description": "US equity fund",
        "currency": "SEK",
        "risk_level": "medium", 
        "expense_ratio": 0.25,
        "category": "us_equity"
    },
    "STOREBRAND_JAPAN_A_SEK": {
        "name": "Storebrand Japan A SEK",
        "type": "japanese_equity",
        "description": "Japanese equity fund",
        "currency": "SEK",
        "risk_level": "medium",
        "expense_ratio": 0.85,
        "category": "japanese_equity"
    },
    "HANDELSBANKEN_GLOBAL_SMAB_INDEX": {
        "name": "Handelsbanken Global småb index",
        "type": "small_cap_global",
        "description": "Global small cap index fund",
        "currency": "SEK",
        "risk_level": "high",
        "expense_ratio": 0.40,
        "category": "small_cap"
    },
    "XETRA_GOLD_ETC": {
        "name": "Xetra-Gold ETC",
        "type": "commodity",
        "description": "Physical gold ETC",
        "currency": "SEK",
        "risk_level": "medium",
        "expense_ratio": 0.35,
        "category": "precious_metals"
    },
    "VIRTUNE_BITCOIN_PRIME_ETP": {
        "name": "Virtune Bitcoin Prime ETP",
        "type": "cryptocurrency",
        "description": "Bitcoin ETP",
        "currency": "SEK",
        "risk_level": "very_high",
        "expense_ratio": 1.00,
        "category": "cryptocurrency"
    },
    "XBT_ETHER_ONE": {
        "name": "XBT Ether One",
        "type": "cryptocurrency", 
        "description": "Ethereum ETP",
        "currency": "SEK",
        "risk_level": "very_high",
        "expense_ratio": 1.00,
        "category": "cryptocurrency"
    },
    "PLUS_FASTIGHETER_SVERIGE_INDEX": {
        "name": "Plus Fastigheter Sverige Index",
        "type": "real_estate",
        "description": "Swedish real estate index fund",
        "currency": "SEK",
        "risk_level": "medium",
        "expense_ratio": 0.30,
        "category": "real_estate"
    }
}


class InvestmentMCPSettings(BaseSettings):
    """Unified settings for Investment MCP System."""
    
    # Application
    APP_NAME: str = "Investment MCP System"
    VERSION: str = "4.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Security
    SECRET_KEY: str = Field(default="dev-secret-key", env="SECRET_KEY")
    API_KEY_HEADER: str = "X-API-Key"
    ALLOWED_ORIGINS: List[str] = Field(default=["http://localhost:3000"], env="ALLOWED_ORIGINS")
    
    # Database
    DATABASE_URL: str = Field(default="sqlite:///./investment_data.db", env="DATABASE_URL")
    DATABASE_POOL_SIZE: int = Field(default=20, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")
    
    # External APIs
    RIKSBANK_API_BASE: str = Field(default="https://api.riksbank.se", env="RIKSBANK_API_BASE")
    SCB_API_BASE: str = Field(default="https://api.scb.se", env="SCB_API_BASE")
    YAHOO_FINANCE_TIMEOUT: int = Field(default=30, env="YAHOO_FINANCE_TIMEOUT")
    
    # AI Configuration
    AI_PROVIDER: str = Field(default="ollama", env="AI_PROVIDER")
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    AI_MODEL_NAME: str = Field(default="llama2", env="AI_MODEL_NAME")
    AI_TIMEOUT_SECONDS: int = Field(default=60, env="AI_TIMEOUT_SECONDS")
    
    # Data Collection
    HISTORICAL_DATA_YEARS: int = Field(default=20, env="HISTORICAL_DATA_YEARS")
    RECENT_DATA_DAYS: int = Field(default=30, env="RECENT_DATA_DAYS")
    DATA_QUALITY_THRESHOLD: float = Field(default=0.9, env="DATA_QUALITY_THRESHOLD")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")
    RATE_LIMIT_BURST: int = Field(default=10, env="RATE_LIMIT_BURST")
    
    # Caching
    CACHE_TTL_SECONDS: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    CACHE_TTL_ECONOMIC_DATA: int = Field(default=1800, env="CACHE_TTL_ECONOMIC_DATA")
    CACHE_TTL_FUND_DATA: int = Field(default=300, env="CACHE_TTL_FUND_DATA")
    
    # Performance
    MAX_CONCURRENT_REQUESTS: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    REQUEST_TIMEOUT_SECONDS: int = Field(default=30, env="REQUEST_TIMEOUT_SECONDS")
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        if v not in ["development", "staging", "production"]:
            raise ValueError("ENVIRONMENT must be one of: development, staging, production")
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        if v.upper() not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("LOG_LEVEL must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
        return v.upper()
    
    @validator("AI_PROVIDER")
    def validate_ai_provider(cls, v):
        if v not in ["openai", "anthropic", "ollama"]:
            raise ValueError("AI_PROVIDER must be one of: openai, anthropic, ollama")
        return v
    
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"
    
    @property
    def database_config(self) -> dict:
        """Get database configuration."""
        return {
            "url": self.DATABASE_URL,
            "pool_size": self.DATABASE_POOL_SIZE,
            "max_overflow": self.DATABASE_MAX_OVERFLOW,
            "echo": self.is_development,
        }
    
    @property
    def ai_config(self) -> dict:
        """Get AI provider configuration."""
        config = {
            "provider": self.AI_PROVIDER,
            "model_name": self.AI_MODEL_NAME,
            "timeout": self.AI_TIMEOUT_SECONDS,
        }
        
        if self.AI_PROVIDER == "openai":
            config["api_key"] = self.OPENAI_API_KEY
        elif self.AI_PROVIDER == "anthropic":
            config["api_key"] = self.ANTHROPIC_API_KEY
        elif self.AI_PROVIDER == "ollama":
            config["base_url"] = self.OLLAMA_BASE_URL
        
        return config
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env


@lru_cache()
def get_settings() -> InvestmentMCPSettings:
    """Get cached settings instance."""
    return InvestmentMCPSettings()


def get_approved_funds() -> List[str]:
    """Return list of all approved fund identifiers."""
    return list(TRADEABLE_FUNDS.keys())


def get_fund_info(fund_id: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific fund."""
    return TRADEABLE_FUNDS.get(fund_id)


def validate_fund_allocation(allocation_dict: Dict[str, float]) -> Dict[str, Any]:
    """
    Validate that allocation only uses approved funds and sums to 100%.
    
    Args:
        allocation_dict: Dictionary of fund_id -> percentage (as decimal)
        
    Returns:
        dict: Validation result with 'valid' boolean and 'errors' list
    """
    errors = []
    
    # Check if all funds are approved
    for fund_id in allocation_dict.keys():
        if fund_id not in TRADEABLE_FUNDS:
            errors.append(f"Fund '{fund_id}' is not in approved universe")
    
    # Check if allocations sum to 100% (allowing 0.1% tolerance)
    total_allocation = sum(allocation_dict.values())
    if abs(total_allocation - 1.0) > 0.001:
        errors.append(f"Allocations sum to {total_allocation:.1%}, must sum to 100%")
    
    # Check for negative allocations
    for fund_id, allocation in allocation_dict.items():
        if allocation < 0:
            errors.append(f"Negative allocation for {fund_id}: {allocation:.1%}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "total_allocation": total_allocation
    }


def get_fund_universe() -> Dict[str, List[str]]:
    """Get the fund universe mapping."""
    return FUND_UNIVERSE


def get_environment_info() -> Dict[str, Any]:
    """Get environment information for debugging."""
    settings = get_settings()
    
    return {
        "app_name": settings.APP_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "debug": settings.DEBUG,
        "ai_provider": settings.AI_PROVIDER,
        "database_configured": bool(settings.DATABASE_URL),
        "rate_limit": settings.RATE_LIMIT_PER_MINUTE,
        "fund_count": len(TRADEABLE_FUNDS),
        "historical_years": settings.HISTORICAL_DATA_YEARS,
    }