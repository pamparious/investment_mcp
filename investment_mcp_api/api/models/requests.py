"""Request models for Investment MCP API."""

from typing import Dict, List, Optional, Any
from datetime import date
from enum import Enum

from pydantic import BaseModel, Field, validator


class RiskProfile(str, Enum):
    """Investment risk profile levels."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class FundCategory(str, Enum):
    """Fund category types."""
    EQUITY = "equity"
    BOND = "bond"
    MIXED = "mixed"
    ALTERNATIVE = "alternative"
    COMMODITY = "commodity"


class Region(str, Enum):
    """Geographic regions for funds."""
    GLOBAL = "global"
    USA = "usa"
    EUROPE = "europe"
    SWEDEN = "sweden"
    EMERGING = "emerging"


class ComparisonPeriod(str, Enum):
    """Time periods for comparison analysis."""
    ONE_YEAR = "1y"
    THREE_YEARS = "3y"
    FIVE_YEARS = "5y"
    TEN_YEARS = "10y"
    TWENTY_YEARS = "20y"
    MAX = "max"


class MetricType(str, Enum):
    """Performance metrics for analysis."""
    RETURN = "return"
    VOLATILITY = "volatility"
    SHARPE = "sharpe"
    MAX_DRAWDOWN = "max_drawdown"
    ALL = "all"


class StressTestScenario(str, Enum):
    """Stress test scenarios."""
    CRISIS_2008 = "2008_crisis"
    COVID_2020 = "covid_2020"
    DOTCOM_2000 = "dotcom_2000"
    ALL = "all"


# Portfolio Analysis Requests
class PortfolioConstraints(BaseModel):
    """Portfolio construction constraints."""
    max_funds: int = Field(default=8, ge=1, le=12, description="Maximum number of funds in portfolio")
    min_allocation_per_fund: float = Field(default=0.05, ge=0.01, le=0.5, description="Minimum allocation per fund")
    exclude_funds: Optional[List[str]] = Field(default=None, description="Funds to exclude from allocation")
    max_single_allocation: float = Field(default=0.4, ge=0.1, le=1.0, description="Maximum allocation to single fund")


class PortfolioAnalysisRequest(BaseModel):
    """Request for comprehensive portfolio analysis."""
    risk_profile: RiskProfile = Field(description="Investor risk tolerance level")
    investment_amount: float = Field(ge=10000, le=100000000, description="Investment amount in SEK")
    investment_horizon_years: int = Field(default=10, ge=1, le=50, description="Investment time horizon in years")
    current_allocation: Optional[Dict[str, float]] = Field(
        default=None, 
        description="Current portfolio allocation (fund_code -> weight)"
    )
    constraints: Optional[PortfolioConstraints] = Field(default=None, description="Portfolio constraints")
    include_stress_test: bool = Field(default=True, description="Include stress testing in analysis")
    include_historical_analysis: bool = Field(default=True, description="Include historical performance analysis")
    
    @validator("current_allocation")
    def validate_allocation_weights(cls, v):
        if v is not None:
            total_weight = sum(v.values())
            if not (0.95 <= total_weight <= 1.05):  # Allow small rounding errors
                raise ValueError("Current allocation weights must sum to approximately 1.0")
            for weight in v.values():
                if not (0 <= weight <= 1):
                    raise ValueError("All allocation weights must be between 0 and 1")
        return v


class HistoricalAllocationRequest(BaseModel):
    """Request for historical allocation analysis."""
    risk_profile: RiskProfile = Field(description="Investor risk tolerance level")
    investment_horizon_years: int = Field(ge=1, le=50, description="Investment time horizon in years")
    historical_years: int = Field(default=20, ge=5, le=25, description="Years of historical data to analyze")
    include_stress_test: bool = Field(default=True, description="Include stress testing")
    include_regime_analysis: bool = Field(default=True, description="Include market regime analysis")
    constraints: Optional[PortfolioConstraints] = Field(default=None, description="Portfolio constraints")


class StressTestRequest(BaseModel):
    """Request for portfolio stress testing."""
    allocation: Dict[str, float] = Field(description="Portfolio allocation to test")
    scenarios: List[StressTestScenario] = Field(
        default=[StressTestScenario.ALL], 
        description="Stress test scenarios to run"
    )
    confidence_levels: List[float] = Field(
        default=[0.95, 0.99], 
        description="Confidence levels for VaR calculations"
    )
    
    @validator("allocation")
    def validate_allocation(cls, v):
        total_weight = sum(v.values())
        if not (0.95 <= total_weight <= 1.05):
            raise ValueError("Allocation weights must sum to approximately 1.0")
        return v


# Fund Data Requests
class FundComparisonRequest(BaseModel):
    """Request for fund performance comparison."""
    fund_codes: List[str] = Field(min_items=2, max_items=12, description="Fund codes to compare")
    comparison_period: ComparisonPeriod = Field(default=ComparisonPeriod.FIVE_YEARS, description="Analysis period")
    metrics: List[MetricType] = Field(default=[MetricType.ALL], description="Metrics to include")
    include_correlations: bool = Field(default=True, description="Include correlation analysis")
    benchmark_fund: Optional[str] = Field(default=None, description="Benchmark fund for relative comparison")


# AI Analysis Requests
class InvestmentRecommendationRequest(BaseModel):
    """Request for AI investment recommendation."""
    risk_profile: RiskProfile = Field(description="Investor risk tolerance")
    amount: float = Field(ge=10000, le=100000000, description="Investment amount in SEK")
    investment_horizon: int = Field(default=10, ge=1, le=50, description="Investment horizon in years")
    current_holdings: Optional[Dict[str, float]] = Field(
        default=None, 
        description="Current holdings (fund_code -> amount in SEK)"
    )
    preferences: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Investment preferences and constraints"
    )
    include_detailed_report: bool = Field(default=True, description="Include detailed AI-generated report")


# Market Analysis Requests
class MarketAnalysisRequest(BaseModel):
    """Request for market analysis."""
    analysis_type: str = Field(description="Type of market analysis")
    date_range: Optional[Dict[str, date]] = Field(default=None, description="Date range for analysis")
    include_sentiment: bool = Field(default=True, description="Include sentiment analysis")
    include_economic_context: bool = Field(default=True, description="Include Swedish economic context")


# System Requests
class HealthCheckRequest(BaseModel):
    """Request for system health check."""
    include_detailed_status: bool = Field(default=False, description="Include detailed component status")
    check_external_services: bool = Field(default=True, description="Check external service connectivity")


class DataRefreshRequest(BaseModel):
    """Request for data refresh."""
    refresh_type: str = Field(description="Type of data to refresh")
    force_refresh: bool = Field(default=False, description="Force refresh even if data is recent")
    notify_completion: bool = Field(default=True, description="Send notification when complete")


# Validation Models
class FundCodeValidation(BaseModel):
    """Validation model for fund codes."""
    fund_code: str = Field(min_length=3, max_length=50, description="Fund identifier")
    
    @validator("fund_code")
    def validate_fund_code_format(cls, v):
        # Valid fund codes are alphanumeric with underscores
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Fund code must contain only alphanumeric characters, underscores, and hyphens")
        return v.upper()


class DateRangeValidation(BaseModel):
    """Validation model for date ranges."""
    start_date: date = Field(description="Start date")
    end_date: date = Field(description="End date")
    
    @validator("end_date")
    def validate_date_range(cls, v, values):
        if "start_date" in values and v <= values["start_date"]:
            raise ValueError("End date must be after start date")
        return v


class AmountValidation(BaseModel):
    """Validation model for investment amounts."""
    amount: float = Field(ge=1000, le=1000000000, description="Amount in SEK")
    currency: str = Field(default="SEK", description="Currency code")
    
    @validator("currency")
    def validate_currency(cls, v):
        if v.upper() != "SEK":
            raise ValueError("Only SEK amounts are currently supported")
        return v.upper()


# Batch Request Models
class BatchPortfolioAnalysisRequest(BaseModel):
    """Request for batch portfolio analysis."""
    requests: List[PortfolioAnalysisRequest] = Field(
        max_items=10, 
        description="List of portfolio analysis requests"
    )
    parallel_execution: bool = Field(default=True, description="Execute requests in parallel")


class BatchFundComparisonRequest(BaseModel):
    """Request for batch fund comparison."""
    comparisons: List[FundComparisonRequest] = Field(
        max_items=5,
        description="List of fund comparison requests"
    )
    aggregate_results: bool = Field(default=False, description="Aggregate results across comparisons")