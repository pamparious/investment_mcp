"""Portfolio optimization models using Pydantic and SQLAlchemy."""

from datetime import datetime
from typing import Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, validator
from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, String, Text, 
    Boolean, JSON, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from .fund import Base


class RiskProfile(str, Enum):
    """Risk profile enumeration."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class OptimizationMethod(str, Enum):
    """Portfolio optimization method enumeration."""
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    GENETIC_ALGORITHM = "genetic"
    EQUAL_WEIGHT = "equal_weight"


class Portfolio(Base):
    """Portfolio SQLAlchemy model."""
    
    __tablename__ = "portfolios"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    user_id = Column(String(255), nullable=True, index=True)
    risk_profile = Column(String(20), nullable=False)
    investment_amount = Column(Float, nullable=False)
    investment_horizon_years = Column(Integer, nullable=False)
    optimization_method = Column(String(50), nullable=False)
    
    # Portfolio metrics
    expected_return = Column(Float, nullable=True)
    expected_volatility = Column(Float, nullable=True)
    expected_sharpe = Column(Float, nullable=True)
    expected_max_drawdown = Column(Float, nullable=True)
    
    # Optimization results
    allocations = Column(JSON, nullable=False)  # {fund_name: weight}
    optimization_score = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Constraints used
    constraints = Column(JSON, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    rebalancing_events = relationship(
        "PortfolioRebalancing", back_populates="portfolio", cascade="all, delete-orphan"
    )
    performance_records = relationship(
        "PortfolioPerformance", back_populates="portfolio", cascade="all, delete-orphan"
    )
    
    __table_args__ = (
        Index('idx_portfolio_user_active', 'user_id', 'is_active'),
        Index('idx_portfolio_risk_profile', 'risk_profile'),
    )


class PortfolioRebalancing(Base):
    """Portfolio rebalancing event SQLAlchemy model."""
    
    __tablename__ = "portfolio_rebalancing"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    rebalancing_date = Column(DateTime, nullable=False, index=True)
    
    # Previous and new allocations
    old_allocations = Column(JSON, nullable=False)
    new_allocations = Column(JSON, nullable=False)
    
    # Rebalancing metrics
    turnover = Column(Float, nullable=True)  # Portfolio turnover percentage
    transaction_costs = Column(Float, nullable=True)
    
    # Reason for rebalancing
    rebalancing_reason = Column(String(100), nullable=False)
    trigger_threshold = Column(Float, nullable=True)
    
    # Performance impact
    expected_return_change = Column(Float, nullable=True)
    expected_risk_change = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="rebalancing_events")
    
    __table_args__ = (
        Index('idx_rebalancing_portfolio_date', 'portfolio_id', 'rebalancing_date'),
    )


class PortfolioPerformance(Base):
    """Portfolio performance tracking SQLAlchemy model."""
    
    __tablename__ = "portfolio_performance"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    
    # Performance metrics
    portfolio_value = Column(Float, nullable=False)
    daily_return = Column(Float, nullable=True)
    cumulative_return = Column(Float, nullable=True)
    
    # Risk metrics
    volatility_30d = Column(Float, nullable=True)
    sharpe_30d = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    var_95 = Column(Float, nullable=True)
    
    # Attribution analysis
    fund_contributions = Column(JSON, nullable=True)  # {fund_name: contribution}
    
    # Benchmark comparison
    benchmark_return = Column(Float, nullable=True)
    alpha = Column(Float, nullable=True)
    beta = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="performance_records")
    
    __table_args__ = (
        Index('idx_performance_portfolio_date', 'portfolio_id', 'date'),
    )


# Pydantic Models for API

class PortfolioConstraints(BaseModel):
    """Portfolio construction constraints."""
    
    max_funds: int = Field(default=8, ge=1, le=12)
    min_allocation: float = Field(default=0.05, ge=0.01, le=0.5)
    max_allocation: float = Field(default=0.4, ge=0.1, le=1.0)
    max_equity_allocation: Optional[float] = Field(None, ge=0, le=1)
    min_bond_allocation: Optional[float] = Field(None, ge=0, le=1)
    exclude_funds: Optional[List[str]] = Field(default_factory=list)
    include_funds: Optional[List[str]] = Field(default_factory=list)
    rebalancing_threshold: float = Field(default=0.05, ge=0.01, le=0.3)
    
    @validator("max_allocation")
    def validate_max_allocation(cls, v, values):
        """Ensure max allocation is greater than min allocation."""
        if "min_allocation" in values and v <= values["min_allocation"]:
            raise ValueError("Max allocation must be greater than min allocation")
        return v


class PortfolioOptimizationRequest(BaseModel):
    """Portfolio optimization request."""
    
    risk_profile: RiskProfile
    investment_amount: float = Field(..., gt=0, le=100_000_000)
    investment_horizon_years: int = Field(..., ge=1, le=50)
    optimization_method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE
    constraints: Optional[PortfolioConstraints] = None
    current_allocations: Optional[Dict[str, float]] = None
    rebalancing: bool = Field(default=False)
    
    # Economic scenario inputs
    expected_market_return: Optional[float] = None
    expected_volatility: Optional[float] = None
    correlation_adjustments: Optional[Dict[str, float]] = None
    
    @validator("current_allocations")
    def validate_current_allocations(cls, v):
        """Validate current allocations sum to approximately 1."""
        if v is not None:
            total = sum(v.values())
            if not (0.95 <= total <= 1.05):
                raise ValueError("Current allocations must sum to approximately 1.0")
        return v


class PortfolioAllocation(BaseModel):
    """Portfolio allocation model."""
    
    fund_name: str
    allocation_weight: float = Field(..., ge=0, le=1)
    investment_amount: float = Field(..., ge=0)
    expected_return: Optional[float] = None
    risk_contribution: Optional[float] = None
    diversification_ratio: Optional[float] = None


class PortfolioMetrics(BaseModel):
    """Portfolio performance metrics."""
    
    expected_annual_return: float
    expected_volatility: float
    expected_sharpe_ratio: float
    expected_max_drawdown: float
    expected_var_95: float
    diversification_ratio: float
    
    # Risk decomposition
    systematic_risk: Optional[float] = None
    idiosyncratic_risk: Optional[float] = None
    concentration_risk: Optional[float] = None
    
    # Performance attribution
    return_attribution: Optional[Dict[str, float]] = None
    risk_attribution: Optional[Dict[str, float]] = None


class OptimizationResult(BaseModel):
    """Portfolio optimization result."""
    
    portfolio_id: Optional[str] = None
    allocations: List[PortfolioAllocation]
    portfolio_metrics: PortfolioMetrics
    optimization_method: OptimizationMethod
    optimization_score: float
    confidence_score: float = Field(..., ge=0, le=1)
    
    # Optimization details
    convergence_achieved: bool
    iterations_used: int
    optimization_time_seconds: float
    
    # Alternative allocations
    alternative_allocations: Optional[List[Dict[str, float]]] = None
    efficient_frontier: Optional[List[Dict[str, float]]] = None


class StressTestScenario(BaseModel):
    """Stress test scenario definition."""
    
    name: str
    description: str
    market_shock: Dict[str, float]  # {asset_class: shock_percentage}
    correlation_changes: Optional[Dict[str, float]] = None
    duration_days: Optional[int] = None


class StressTestResult(BaseModel):
    """Stress test result for a portfolio."""
    
    scenario_name: str
    portfolio_return: float
    portfolio_value_change: float
    max_drawdown: float
    recovery_time_days: Optional[int] = None
    
    # Fund-level impacts
    fund_impacts: Dict[str, float]
    worst_performing_fund: str
    best_performing_fund: str
    
    # Risk metrics during stress
    volatility_during_stress: float
    var_95_during_stress: float


class PortfolioStressTest(BaseModel):
    """Complete portfolio stress test results."""
    
    portfolio_id: str
    stress_test_date: datetime
    scenarios: List[StressTestResult]
    
    # Summary metrics
    worst_case_loss: float
    best_case_gain: float
    average_loss: float
    scenarios_with_positive_return: int
    
    # Risk recommendations
    recommended_hedges: Optional[List[str]] = None
    risk_reduction_suggestions: Optional[List[str]] = None


class BacktestResult(BaseModel):
    """Portfolio backtest result."""
    
    portfolio_id: str
    backtest_period_start: datetime
    backtest_period_end: datetime
    
    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Risk metrics
    var_95: float
    cvar_95: float
    downside_deviation: float
    
    # Benchmark comparison
    benchmark_return: float
    alpha: float
    beta: float
    information_ratio: float
    tracking_error: float
    
    # Time series data
    portfolio_values: List[Dict[str, Union[datetime, float]]]
    drawdown_series: List[Dict[str, Union[datetime, float]]]
    
    # Attribution analysis
    return_attribution: Dict[str, float]
    performance_attribution: Dict[str, Dict[str, float]]


class RebalancingRecommendation(BaseModel):
    """Portfolio rebalancing recommendation."""
    
    portfolio_id: str
    current_allocations: Dict[str, float]
    target_allocations: Dict[str, float]
    recommended_trades: Dict[str, float]  # {fund: trade_amount}
    
    # Rebalancing metrics
    portfolio_drift: float
    turnover_required: float
    estimated_transaction_costs: float
    
    # Impact analysis
    expected_return_improvement: float
    expected_risk_reduction: float
    
    # Timing recommendation
    urgency_score: float = Field(..., ge=0, le=1)
    recommended_action: str  # "immediate", "within_week", "within_month", "monitor"
    
    # Justification
    rebalancing_reasons: List[str]
    market_conditions: Optional[str] = None


class PortfolioSummary(BaseModel):
    """Portfolio summary for dashboard display."""
    
    portfolio_id: str
    name: str
    risk_profile: RiskProfile
    total_value: float
    
    # Performance (last update)
    current_return_1d: float
    current_return_1m: float
    current_return_3m: float
    current_return_1y: float
    
    # Risk metrics
    current_volatility: float
    current_sharpe: float
    current_max_drawdown: float
    
    # Allocation summary
    top_holdings: List[Dict[str, Union[str, float]]]  # [{name, weight}]
    asset_class_breakdown: Dict[str, float]
    geographic_breakdown: Dict[str, float]
    
    # Status
    last_rebalanced: Optional[datetime] = None
    next_review_date: Optional[datetime] = None
    health_score: float = Field(..., ge=0, le=1)
    
    # Alerts
    rebalancing_needed: bool
    performance_alerts: List[str]
    risk_alerts: List[str]