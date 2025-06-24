import os
from dotenv import load_dotenv
from typing import List
import logging

load_dotenv()

def safe_int(value: str, default: int) -> int:
    """Safely convert string to int, handling comments and errors."""
    if not value:
        return default
    
    # Remove inline comments
    value = value.split('#')[0].strip()
    
    try:
        return int(value)
    except ValueError:
        logging.warning(f"Could not convert '{value}' to int, using default: {default}")
        return default

def safe_bool(value: str, default: bool = False) -> bool:
    """Safely convert string to bool."""
    if not value:
        return default
    
    # Remove inline comments
    value = value.split('#')[0].strip().lower()
    
    return value in ('true', '1', 'yes', 'on')

class Settings:
    # Existing API settings
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./investment_data.db")
    MCP_SERVER_PORT = safe_int(os.getenv("MCP_SERVER_PORT", "8000"), 8000)
    MCP_LOG_LEVEL = os.getenv("MCP_LOG_LEVEL", "INFO")
    
    # Swedish APIs
    RIKSBANKEN_API_BASE = os.getenv("RIKSBANKEN_API_BASE", "https://api.riksbank.se")
    RIKSBANKEN_API_KEY = os.getenv("RIKSBANKEN_API_KEY")
    SCB_API_BASE = os.getenv("SCB_API_BASE", "https://api.scb.se")
    SCB_API_KEY = os.getenv("SCB_API_KEY")
    
    # Data Collection Settings - using safe_int to handle comments
    DATA_COLLECTION_INTERVAL = safe_int(os.getenv("DATA_COLLECTION_INTERVAL", "3600"), 3600)
    MAX_API_CALLS_PER_MINUTE = safe_int(os.getenv("MAX_API_CALLS_PER_MINUTE", "60"), 60)
    CACHE_DURATION = safe_int(os.getenv("CACHE_DURATION", "1800"), 1800)
    
    # Database Configuration
    DB_POOL_SIZE = safe_int(os.getenv("DB_POOL_SIZE", "5"), 5)
    DB_MAX_OVERFLOW = safe_int(os.getenv("DB_MAX_OVERFLOW", "10"), 10)
    
    # Error Handling
    MAX_RETRIES = safe_int(os.getenv("MAX_RETRIES", "3"), 3)
    RETRY_DELAY = safe_int(os.getenv("RETRY_DELAY", "5"), 5)
    REQUEST_TIMEOUT = safe_int(os.getenv("REQUEST_TIMEOUT", "30"), 30)
    
    # Development
    DEBUG = safe_bool(os.getenv("DEBUG", "False"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Default symbols to track
    DEFAULT_STOCK_SYMBOLS = [
        "^OMX",      # OMX Stockholm
        "^GSPC",     # S&P 500
        "^IXIC",     # NASDAQ
        "^FTSE",     # FTSE 100
        "^GDAXI",    # DAX
    ]
    
    # Swedish housing market regions
    SCB_HOUSING_REGIONS = [
        "00",  # Whole country
        "01",  # Stockholm
        "03",  # Malmö
        "14",  # Göteborg
    ]

settings = Settings()

# Validation on import
if __name__ == "__main__":
    print("Settings validation:")
    print(f"DATA_COLLECTION_INTERVAL: {settings.DATA_COLLECTION_INTERVAL}")
    print(f"MAX_API_CALLS_PER_MINUTE: {settings.MAX_API_CALLS_PER_MINUTE}")
    print(f"CACHE_DURATION: {settings.CACHE_DURATION}")
    print(f"DEBUG: {settings.DEBUG}")