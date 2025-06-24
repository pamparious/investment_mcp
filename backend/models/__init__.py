"""SQLAlchemy database models for investment data."""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

# Create the declarative base
Base = declarative_base()

class MarketData(Base):
    """Stock market data model."""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<MarketData(symbol={self.symbol}, date={self.date}, close={self.close_price})>"

class RiksbankData(Base):
    """Riksbank economic data model."""
    __tablename__ = "riksbank_data"
    
    id = Column(Integer, primary_key=True, index=True)
    series_id = Column(String(50), index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)
    value = Column(Float, nullable=False)
    description = Column(String(200))
    unit = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<RiksbankData(series={self.series_id}, date={self.date}, value={self.value})>"

class SCBData(Base):
    """SCB (Statistics Sweden) data model."""
    __tablename__ = "scb_data"
    
    id = Column(Integer, primary_key=True, index=True)
    table_id = Column(String(50), index=True, nullable=False)
    region = Column(String(10), index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)
    value = Column(Float, nullable=False)
    description = Column(String(200))
    unit = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<SCBData(table={self.table_id}, region={self.region}, date={self.date}, value={self.value})>"

class DataCollectionLog(Base):
    """Data collection activity log model."""
    __tablename__ = "data_collection_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(50), index=True, nullable=False)
    collection_type = Column(String(50), nullable=False)
    status = Column(String(20), index=True, nullable=False)  # success, error
    records_collected = Column(Integer, default=0)
    error_message = Column(Text)
    started_at = Column(DateTime, index=True, nullable=False)
    completed_at = Column(DateTime)
    extra_data = Column(Text)  # JSON string for additional data
    
    def __repr__(self):
        return f"<DataCollectionLog(source={self.source}, status={self.status}, records={self.records_collected})>"

class AnalysisResult(Base):
    """Analysis results model."""
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_type = Column(String(50), index=True, nullable=False)
    symbol = Column(String(20), index=True)
    region = Column(String(10), index=True)
    result_data = Column(Text, nullable=False)  # JSON string
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    extra_data = Column(Text)  # JSON string for additional data
    
    def __repr__(self):
        return f"<AnalysisResult(type={self.analysis_type}, symbol={self.symbol}, confidence={self.confidence_score})>"

class Portfolio(Base):
    """Portfolio model."""
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    total_value = Column(Float, default=0.0)
    cash_balance = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to holdings
    holdings = relationship("PortfolioHolding", back_populates="portfolio")
    
    def __repr__(self):
        return f"<Portfolio(name={self.name}, value={self.total_value})>"

class PortfolioHolding(Base):
    """Portfolio holdings model."""
    __tablename__ = "portfolio_holdings"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'), nullable=False)
    symbol = Column(String(20), index=True, nullable=False)
    shares = Column(Float, nullable=False)
    average_cost = Column(Float, nullable=False)
    current_price = Column(Float)
    current_value = Column(Float)
    unrealized_gain_loss = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to portfolio
    portfolio = relationship("Portfolio", back_populates="holdings")
    
    def __repr__(self):
        return f"<PortfolioHolding(symbol={self.symbol}, shares={self.shares}, value={self.current_value})>"

# Export all models for easy importing
__all__ = [
    'Base',
    'MarketData', 
    'RiksbankData',
    'SCBData',
    'DataCollectionLog',
    'AnalysisResult',
    'Portfolio',
    'PortfolioHolding'
]