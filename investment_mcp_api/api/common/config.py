"""Configuration management for Investment MCP API."""

import os
from typing import List, Optional
from functools import lru_cache

from pydantic import BaseSettings, validator, Field


class Settings(BaseSettings):
    """Application settings using Pydantic BaseSettings."""
    
    # Application
    APP_NAME: str = "Investment MCP API"
    VERSION: str = "3.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Security
    SECRET_KEY: str = Field(env="SECRET_KEY")
    API_KEY_HEADER: str = "X-API-Key"
    ALLOWED_ORIGINS: List[str] = Field(default=["http://localhost:3000"], env="ALLOWED_ORIGINS")
    ALLOWED_HOSTS: List[str] = Field(default=["localhost", "127.0.0.1"], env="ALLOWED_HOSTS")
    
    # Database
    DATABASE_URL: str = Field(env="DATABASE_URL")
    DATABASE_POOL_SIZE: int = Field(default=20, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")
    
    # Redis Cache
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    CACHE_TTL_SECONDS: int = Field(default=3600, env="CACHE_TTL_SECONDS")  # 1 hour
    CACHE_TTL_ECONOMIC_DATA: int = Field(default=1800, env="CACHE_TTL_ECONOMIC_DATA")  # 30 minutes
    CACHE_TTL_FUND_DATA: int = Field(default=300, env="CACHE_TTL_FUND_DATA")  # 5 minutes
    
    # External APIs
    RIKSBANK_API_BASE: str = Field(default="https://api.riksbank.se", env="RIKSBANK_API_BASE")
    SCB_API_BASE: str = Field(default="https://api.scb.se", env="SCB_API_BASE")
    YAHOO_FINANCE_TIMEOUT: int = Field(default=30, env="YAHOO_FINANCE_TIMEOUT")
    
    # AI Configuration
    AI_PROVIDER: str = Field(default="ollama", env="AI_PROVIDER")
    OPENAI_API_KEY: Optional[str] = Field(env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(env="ANTHROPIC_API_KEY")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    AI_MODEL_NAME: str = Field(default="llama2", env="AI_MODEL_NAME")
    AI_TIMEOUT_SECONDS: int = Field(default=60, env="AI_TIMEOUT_SECONDS")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")
    RATE_LIMIT_PREMIUM_PER_MINUTE: int = Field(default=1000, env="RATE_LIMIT_PREMIUM_PER_MINUTE")
    RATE_LIMIT_BURST: int = Field(default=10, env="RATE_LIMIT_BURST")
    
    # Data Collection
    HISTORICAL_DATA_YEARS: int = Field(default=20, env="HISTORICAL_DATA_YEARS")
    RECENT_DATA_DAYS: int = Field(default=30, env="RECENT_DATA_DAYS")
    DATA_QUALITY_THRESHOLD: float = Field(default=0.9, env="DATA_QUALITY_THRESHOLD")
    
    # File Storage
    DATA_STORAGE_PATH: str = Field(default="./storage/data", env="DATA_STORAGE_PATH")
    CACHE_STORAGE_PATH: str = Field(default="./storage/cache", env="CACHE_STORAGE_PATH")
    LOGS_STORAGE_PATH: str = Field(default="./storage/logs", env="LOGS_STORAGE_PATH")
    
    # Performance
    MAX_CONCURRENT_REQUESTS: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    REQUEST_TIMEOUT_SECONDS: int = Field(default=30, env="REQUEST_TIMEOUT_SECONDS")
    BACKGROUND_TASK_TIMEOUT: int = Field(default=300, env="BACKGROUND_TASK_TIMEOUT")
    
    # Monitoring
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    HEALTH_CHECK_INTERVAL: int = Field(default=60, env="HEALTH_CHECK_INTERVAL")
    
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
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("ALLOWED_HOSTS", pre=True)
    def parse_allowed_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
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
    def redis_config(self) -> dict:
        """Get Redis configuration."""
        return {
            "url": self.REDIS_URL,
            "encoding": "utf-8",
            "decode_responses": True,
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


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_environment_info() -> dict:
    """Get environment information for debugging."""
    settings = get_settings()
    
    return {
        "app_name": settings.APP_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "debug": settings.DEBUG,
        "ai_provider": settings.AI_PROVIDER,
        "database_configured": bool(settings.DATABASE_URL),
        "redis_configured": bool(settings.REDIS_URL),
        "rate_limit": settings.RATE_LIMIT_PER_MINUTE,
        "data_storage": settings.DATA_STORAGE_PATH,
    }