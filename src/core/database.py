"""
Unified database models for Investment MCP System.

This module consolidates all database models from various scattered model files
into a single, comprehensive database layer using SQLAlchemy.
"""

from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Date, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey

Base = declarative_base()


class MarketData(Base):
    """Market data for funds and securities."""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(50), index=True, nullable=False)
    fund_name = Column(String(200), nullable=False)
    date = Column(Date, index=True, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    adjusted_close = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<MarketData(symbol={self.symbol}, date={self.date}, close={self.close_price})>"


class EconomicData(Base):
    """Swedish economic data from Riksbanken and SCB."""
    __tablename__ = "economic_data"
    
    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(20), index=True, nullable=False)  # 'riksbank' or 'scb'
    series_id = Column(String(50), index=True, nullable=False)
    table_id = Column(String(50), index=True)  # for SCB data
    region = Column(String(50), index=True)  # for regional data
    date = Column(Date, index=True, nullable=False)
    value = Column(Float, nullable=False)
    description = Column(String(500))
    unit = Column(String(50))
    frequency = Column(String(20))  # daily, monthly, quarterly, yearly
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<EconomicData(source={self.source}, series={self.series_id}, date={self.date}, value={self.value})>"


class AnalysisResult(Base):
    """Results from technical and fundamental analysis."""
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_type = Column(String(50), index=True, nullable=False)  # 'technical', 'fundamental', 'portfolio'
    symbol = Column(String(50), index=True)
    fund_name = Column(String(200))
    analysis_date = Column(Date, index=True, nullable=False)
    result_data = Column(JSON, nullable=False)  # JSON data for analysis results
    confidence_score = Column(Float)
    risk_metrics = Column(JSON)  # Risk calculations as JSON
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<AnalysisResult(type={self.analysis_type}, symbol={self.symbol}, confidence={self.confidence_score})>"


class Portfolio(Base):
    """Portfolio definitions and metadata."""
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    portfolio_type = Column(String(50))  # 'conservative', 'balanced', 'growth', 'custom'
    target_return = Column(Float)
    risk_tolerance = Column(String(20))  # 'low', 'medium', 'high', 'very_high'
    total_value = Column(Float, default=0.0)
    cash_balance = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to holdings
    holdings = relationship("PortfolioHolding", back_populates="portfolio")
    
    def __repr__(self):
        return f"<Portfolio(name={self.name}, type={self.portfolio_type}, value={self.total_value})>"


class PortfolioHolding(Base):
    """Individual fund holdings within portfolios."""
    __tablename__ = "portfolio_holdings"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    fund_id = Column(String(50), index=True, nullable=False)  # from TRADEABLE_FUNDS
    fund_name = Column(String(200), nullable=False)
    symbol = Column(String(50), index=True, nullable=False)
    allocation_percentage = Column(Float, nullable=False)  # 0.0 to 1.0
    shares = Column(Float, nullable=False)
    average_cost = Column(Float, nullable=False)
    current_price = Column(Float)
    current_value = Column(Float)
    unrealized_gain_loss = Column(Float)
    unrealized_gain_loss_pct = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to portfolio
    portfolio = relationship("Portfolio", back_populates="holdings")
    
    def __repr__(self):
        return f"<PortfolioHolding(fund={self.fund_name}, allocation={self.allocation_percentage:.1%}, value={self.current_value})>"


class OptimizationResult(Base):
    """Portfolio optimization results and recommendations."""
    __tablename__ = "optimization_results"
    
    id = Column(Integer, primary_key=True, index=True)
    optimization_type = Column(String(50), nullable=False)  # 'mean_variance', 'risk_parity', 'ai_enhanced'
    risk_tolerance = Column(String(20), nullable=False)
    target_return = Column(Float)
    expected_return = Column(Float)
    expected_volatility = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    allocation_weights = Column(JSON, nullable=False)  # fund_id -> weight mapping
    backtest_metrics = Column(JSON)  # Historical performance metrics
    ai_reasoning = Column(Text)  # AI explanation of recommendations
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<OptimizationResult(type={self.optimization_type}, return={self.expected_return:.2%}, sharpe={self.sharpe_ratio:.2f})>"


class DataCollectionLog(Base):
    """Log of data collection activities and status."""
    __tablename__ = "data_collection_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(50), index=True, nullable=False)  # 'yahoo', 'riksbank', 'scb'
    collection_type = Column(String(50), nullable=False)  # 'historical', 'daily_update', 'economic'
    status = Column(String(20), index=True, nullable=False)  # 'success', 'error', 'partial'
    records_collected = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    error_message = Column(Text)
    error_details = Column(JSON)  # Detailed error information
    started_at = Column(DateTime, index=True, nullable=False)
    completed_at = Column(DateTime)
    duration_seconds = Column(Float)
    extra_data = Column(JSON)  # Additional metadata
    
    def __repr__(self):
        return f"<DataCollectionLog(source={self.source}, status={self.status}, records={self.records_collected})>"


class AIAnalysisLog(Base):
    """Log of AI analysis requests and responses."""
    __tablename__ = "ai_analysis_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_type = Column(String(50), index=True, nullable=False)
    ai_provider = Column(String(20), nullable=False)  # 'openai', 'anthropic', 'ollama'
    model_name = Column(String(50), nullable=False)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)
    response_time_ms = Column(Integer)
    status = Column(String(20), index=True, nullable=False)  # 'success', 'error', 'timeout'
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<AIAnalysisLog(type={self.analysis_type}, provider={self.ai_provider}, tokens={self.total_tokens})>"


class SwedishEconomicIndicator(Base):
    """Specific Swedish economic indicators for investment analysis."""
    __tablename__ = "swedish_economic_indicators"
    
    id = Column(Integer, primary_key=True, index=True)
    indicator_name = Column(String(100), index=True, nullable=False)
    indicator_type = Column(String(50), nullable=False)  # 'interest_rate', 'inflation', 'housing', 'gdp'
    date = Column(Date, index=True, nullable=False)
    value = Column(Float, nullable=False)
    previous_value = Column(Float)
    change_pct = Column(Float)
    unit = Column(String(50))
    frequency = Column(String(20))  # monthly, quarterly
    source = Column(String(50))  # riksbank, scb
    impact_rating = Column(String(20))  # 'low', 'medium', 'high' - impact on investments
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<SwedishEconomicIndicator(name={self.indicator_name}, date={self.date}, value={self.value})>"


def validate_fund_allocation(allocation_dict: Dict[str, float]) -> Dict[str, Any]:
    """
    Validate that allocation only uses approved funds and sums to 100%.
    
    Args:
        allocation_dict: Dictionary of fund_id -> percentage (as decimal)
        
    Returns:
        dict: Validation result with 'valid' boolean and 'errors' list
    """
    from .config import TRADEABLE_FUNDS
    
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