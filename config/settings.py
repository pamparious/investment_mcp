import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./investment_data.db")
    
    # MCP Configuration
    MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", 8000))
    MCP_LOG_LEVEL = os.getenv("MCP_LOG_LEVEL", "INFO")
    
    # Swedish APIs
    RIKSBANKEN_API_BASE = os.getenv("RIKSBANKEN_API_BASE", "https://api.riksbank.se")
    SCB_API_BASE = os.getenv("SCB_API_BASE", "https://api.scb.se")
    
    # Development
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()