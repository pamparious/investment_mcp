"""Fund data models using Pydantic and SQLAlchemy."""

from datetime import date, datetime
from decimal import Decimal
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator
from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, String, Text, Boolean,
    Date, Numeric, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSON, UUID
import uuid

Base = declarative_base()


class Fund(Base):
    """Fund SQLAlchemy model."""
    
    __tablename__ = "funds"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), unique=True, nullable=False, index=True)
    ticker = Column(String(50), unique=True, nullable=True, index=True)
    category = Column(String(100), nullable=False, index=True)
    isin = Column(String(12), nullable=True, unique=True)
    currency = Column(String(3), default="SEK", nullable=False)
    management_fee = Column(Float, nullable=True)
    description = Column(Text, nullable=True)
    inception_date = Column(Date, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    prices = relationship("FundPrice", back_populates="fund", cascade="all, delete-orphan")
    analytics = relationship("FundAnalytics", back_populates="fund", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_fund_category_active', 'category', 'is_active'),
    )


class FundPrice(Base):
    """Fund daily price data SQLAlchemy model."""
    
    __tablename__ = "fund_prices"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    fund_id = Column(UUID(as_uuid=True), ForeignKey("funds.id"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    open_price = Column(Numeric(precision=12, scale=4), nullable=True)
    high_price = Column(Numeric(precision=12, scale=4), nullable=True)
    low_price = Column(Numeric(precision=12, scale=4), nullable=True)
    close_price = Column(Numeric(precision=12, scale=4), nullable=False)
    adjusted_close = Column(Numeric(precision=12, scale=4), nullable=True)
    volume = Column(Integer, nullable=True)
    daily_return = Column(Float, nullable=True)
    data_source = Column(String(50), default="yfinance", nullable=False)
    data_quality = Column(String(20), default="unknown", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    fund = relationship("Fund", back_populates="prices")
    
    __table_args__ = (
        Index('idx_fund_price_date', 'fund_id', 'date'),
        Index('idx_fund_price_date_desc', 'fund_id', 'date'.desc()),
    )


class FundAnalytics(Base):
    """Fund analytics and technical indicators SQLAlchemy model."""
    
    __tablename__ = "fund_analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    fund_id = Column(UUID(as_uuid=True), ForeignKey("funds.id"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    
    # Moving averages
    sma_20 = Column(Float, nullable=True)
    sma_50 = Column(Float, nullable=True)
    sma_200 = Column(Float, nullable=True)
    ema_12 = Column(Float, nullable=True)
    ema_26 = Column(Float, nullable=True)
    
    # Technical indicators
    rsi = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    macd_signal = Column(Float, nullable=True)
    macd_histogram = Column(Float, nullable=True)
    bollinger_upper = Column(Float, nullable=True)
    bollinger_lower = Column(Float, nullable=True)
    bollinger_width = Column(Float, nullable=True)
    
    # Volatility metrics
    volatility_30d = Column(Float, nullable=True)
    volatility_90d = Column(Float, nullable=True)
    volatility_252d = Column(Float, nullable=True)
    
    # Risk metrics
    sharpe_252d = Column(Float, nullable=True)
    var_5_30d = Column(Float, nullable=True)
    var_5_90d = Column(Float, nullable=True)
    max_drawdown_252d = Column(Float, nullable=True)
    
    # Market regime indicators
    is_bull_market = Column(Boolean, nullable=True)
    is_bear_market = Column(Boolean, nullable=True)
    trend_strength = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    fund = relationship("Fund", back_populates="analytics")
    
    __table_args__ = (
        Index('idx_fund_analytics_date', 'fund_id', 'date'),
    )


# Pydantic Models for API

class FundBase(BaseModel):
    """Base fund model."""
    
    name: str = Field(..., min_length=1, max_length=255)
    ticker: Optional[str] = Field(None, max_length=50)
    category: str = Field(..., min_length=1, max_length=100)
    isin: Optional[str] = Field(None, min_length=12, max_length=12)
    currency: str = Field(default="SEK", min_length=3, max_length=3)
    management_fee: Optional[float] = Field(None, ge=0, le=0.1)
    description: Optional[str] = None
    inception_date: Optional[date] = None
    is_active: bool = Field(default=True)
    
    @validator("currency")
    def validate_currency(cls, v):
        """Validate currency code."""
        if v not in ["SEK", "USD", "EUR", "NOK", "DKK"]:
            raise ValueError("Currency must be SEK, USD, EUR, NOK, or DKK")
        return v
    
    @validator("category")
    def validate_category(cls, v):
        """Validate fund category."""
        valid_categories = [
            "global_equity", "emerging_markets", "europe", "nordic",
            "sweden", "usa", "japan", "small_cap", "commodities",
            "crypto", "real_estate"
        ]
        if v not in valid_categories:
            raise ValueError(f"Category must be one of {valid_categories}")
        return v


class FundCreate(FundBase):
    """Fund creation model."""
    pass


class FundUpdate(BaseModel):
    """Fund update model."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    ticker: Optional[str] = Field(None, max_length=50)
    category: Optional[str] = Field(None, min_length=1, max_length=100)
    isin: Optional[str] = Field(None, min_length=12, max_length=12)
    currency: Optional[str] = Field(None, min_length=3, max_length=3)
    management_fee: Optional[float] = Field(None, ge=0, le=0.1)
    description: Optional[str] = None
    inception_date: Optional[date] = None
    is_active: Optional[bool] = None


class FundResponse(FundBase):
    """Fund response model."""
    
    id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        """Pydantic config."""
        from_attributes = True


class FundPriceBase(BaseModel):
    """Base fund price model."""
    
    date: date
    open_price: Optional[Decimal] = Field(None, decimal_places=4)
    high_price: Optional[Decimal] = Field(None, decimal_places=4)
    low_price: Optional[Decimal] = Field(None, decimal_places=4)
    close_price: Decimal = Field(..., decimal_places=4, gt=0)
    adjusted_close: Optional[Decimal] = Field(None, decimal_places=4)
    volume: Optional[int] = Field(None, ge=0)
    daily_return: Optional[float] = None
    data_source: str = Field(default="yfinance", max_length=50)
    data_quality: str = Field(default="unknown", max_length=20)
    
    @validator("data_quality")
    def validate_data_quality(cls, v):
        """Validate data quality."""
        if v not in ["high", "medium", "low", "unknown"]:
            raise ValueError("Data quality must be high, medium, low, or unknown")
        return v


class FundPriceCreate(FundPriceBase):
    """Fund price creation model."""
    fund_id: str


class FundPriceResponse(FundPriceBase):
    """Fund price response model."""
    
    id: str
    fund_id: str
    created_at: datetime
    
    class Config:
        """Pydantic config."""
        from_attributes = True


class FundAnalyticsBase(BaseModel):
    """Base fund analytics model."""
    
    date: date
    
    # Moving averages
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    
    # Technical indicators
    rsi: Optional[float] = Field(None, ge=0, le=100)
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    bollinger_width: Optional[float] = None
    
    # Volatility metrics
    volatility_30d: Optional[float] = Field(None, ge=0)
    volatility_90d: Optional[float] = Field(None, ge=0)
    volatility_252d: Optional[float] = Field(None, ge=0)
    
    # Risk metrics
    sharpe_252d: Optional[float] = None
    var_5_30d: Optional[float] = None
    var_5_90d: Optional[float] = None
    max_drawdown_252d: Optional[float] = Field(None, le=0)
    
    # Market regime indicators
    is_bull_market: Optional[bool] = None
    is_bear_market: Optional[bool] = None
    trend_strength: Optional[float] = None


class FundAnalyticsCreate(FundAnalyticsBase):
    """Fund analytics creation model."""
    fund_id: str


class FundAnalyticsResponse(FundAnalyticsBase):
    """Fund analytics response model."""
    
    id: str
    fund_id: str
    created_at: datetime
    
    class Config:
        """Pydantic config."""
        from_attributes = True


class FundPerformanceMetrics(BaseModel):
    """Fund performance metrics model."""
    
    fund_id: str
    fund_name: str
    period_start: date
    period_end: date
    
    # Return metrics
    total_return: float
    annualized_return: float
    best_year: Optional[float] = None
    worst_year: Optional[float] = None
    positive_years_pct: Optional[float] = None
    
    # Risk metrics
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    
    # Statistical metrics
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # Data quality
    data_points: int
    data_quality_score: float = Field(..., ge=0, le=1)


class FundComparison(BaseModel):
    """Fund comparison model."""
    
    funds: List[FundPerformanceMetrics]
    comparison_period: str
    benchmark_fund: Optional[str] = None
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    
    # Ranking metrics
    best_return: str
    best_sharpe: str
    lowest_volatility: str
    lowest_drawdown: str