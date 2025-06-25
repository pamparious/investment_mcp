"""Response models for Investment MCP API."""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
from enum import Enum

from pydantic import BaseModel, Field


class ResponseStatus(str, Enum):
    """API response status."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


class DataQuality(str, Enum):
    """Data quality levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MarketSentiment(str, Enum):
    """Market sentiment levels."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"


# Base Response Models
class BaseResponse(BaseModel):
    """Base response model with common fields."""
    status: ResponseStatus = Field(description="Response status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(default=None, description="Request tracking ID")


class ErrorResponse(BaseResponse):
    """Error response model."""
    status: ResponseStatus = ResponseStatus.ERROR
    error: str = Field(description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code for client handling")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")


# Portfolio Analysis Responses
class PortfolioMetrics(BaseModel):
    """Portfolio performance metrics."""
    expected_annual_return: float = Field(description="Expected annual return")
    expected_volatility: float = Field(description="Expected annual volatility")
    expected_sharpe_ratio: float = Field(description="Expected Sharpe ratio")
    expected_max_drawdown: float = Field(description="Expected maximum drawdown")
    diversification_ratio: float = Field(description="Portfolio diversification ratio")
    value_at_risk_95: Optional[float] = Field(default=None, description="95% Value at Risk")
    conditional_var_95: Optional[float] = Field(default=None, description="95% Conditional VaR")


class HistoricalPerformance(BaseModel):
    """Historical performance analysis."""
    annualized_return: float = Field(description="Historical annualized return")
    volatility: float = Field(description="Historical volatility")
    sharpe_ratio: float = Field(description="Historical Sharpe ratio")
    max_drawdown: float = Field(description="Historical maximum drawdown")
    best_year_return: float = Field(description="Best calendar year return")
    worst_year_return: float = Field(description="Worst calendar year return")
    positive_years_percentage: float = Field(description="Percentage of positive years")
    total_return: Dict[str, float] = Field(description="Total returns over different periods")


class RegimePerformance(BaseModel):
    """Performance in specific market regime."""
    average_return: float = Field(description="Average return in regime")
    volatility: float = Field(description="Volatility in regime")
    frequency: float = Field(description="Frequency of regime occurrence")
    max_drawdown: float = Field(description="Maximum drawdown in regime")


class RegimeAnalysis(BaseModel):
    """Market regime analysis."""
    bull_market: RegimePerformance = Field(description="Bull market performance")
    bear_market: RegimePerformance = Field(description="Bear market performance")
    high_volatility: RegimePerformance = Field(description="High volatility period performance")
    low_volatility: RegimePerformance = Field(description="Low volatility period performance")
    crisis: RegimePerformance = Field(description="Crisis period performance")


class StressTestScenarioResult(BaseModel):
    """Single stress test scenario result."""
    scenario_name: str = Field(description="Scenario identifier")
    portfolio_return: float = Field(description="Portfolio return during scenario")
    duration_days: int = Field(description="Scenario duration in days")
    max_drawdown: float = Field(description="Maximum drawdown during scenario")
    recovery_time_days: Optional[int] = Field(default=None, description="Recovery time in days")
    description: str = Field(description="Scenario description")


class StressTestResults(BaseModel):
    """Comprehensive stress test results."""
    scenarios: Dict[str, StressTestScenarioResult] = Field(description="Individual scenario results")
    worst_case_loss: float = Field(description="Worst case loss across all scenarios")
    average_crisis_performance: float = Field(description="Average performance in crisis scenarios")
    recovery_time_estimate: str = Field(description="Estimated recovery time")
    risk_score: float = Field(ge=0, le=100, description="Overall risk score (0-100)")


class SwedishEconomicContext(BaseModel):
    """Swedish economic context information."""
    current_phase: str = Field(description="Current economic cycle phase")
    key_indicators: Dict[str, float] = Field(description="Key economic indicators")
    trends: Dict[str, str] = Field(description="Economic trends")
    investment_implications: List[str] = Field(description="Investment implications")
    confidence_score: float = Field(ge=0, le=1, description="Economic analysis confidence")


class AIReasoning(BaseModel):
    """AI-generated reasoning and explanations."""
    allocation_rationale: str = Field(description="Rationale for allocation decisions")
    swedish_economic_rationale: str = Field(description="Swedish economic context reasoning")
    risk_assessment: str = Field(description="Risk assessment explanation")
    historical_context: str = Field(description="Historical context analysis")
    regime_considerations: str = Field(description="Market regime considerations")


class PortfolioAnalysisResponse(BaseResponse):
    """Portfolio analysis response."""
    allocation: Dict[str, float] = Field(description="Recommended allocation")
    expected_metrics: PortfolioMetrics = Field(description="Expected portfolio metrics")
    historical_analysis: HistoricalPerformance = Field(description="Historical performance analysis")
    stress_test: StressTestResults = Field(description="Stress test results")
    regime_analysis: RegimeAnalysis = Field(description="Market regime analysis")
    swedish_economic_context: SwedishEconomicContext = Field(description="Swedish economic context")
    ai_reasoning: AIReasoning = Field(description="AI-generated reasoning")
    confidence_score: float = Field(ge=0, le=1, description="Analysis confidence score")
    data_quality_score: float = Field(ge=0, le=1, description="Underlying data quality score")


class HistoricalAllocationResponse(BaseResponse):
    """Historical allocation analysis response."""
    allocation: Dict[str, float] = Field(description="Optimal historical allocation")
    historical_reasoning: str = Field(description="Historical analysis reasoning")
    performance_analysis: HistoricalPerformance = Field(description="Performance metrics")
    regime_analysis: RegimeAnalysis = Field(description="Regime-based analysis")
    correlation_analysis: Dict[str, Dict[str, float]] = Field(description="Correlation matrix")
    optimization_details: Dict[str, Any] = Field(description="Optimization process details")


class StressTestResponse(BaseResponse):
    """Stress test response."""
    scenarios: Dict[str, StressTestScenarioResult] = Field(description="Scenario results")
    overall_assessment: Dict[str, Any] = Field(description="Overall portfolio assessment")
    recommendations: List[str] = Field(description="Risk management recommendations")
    risk_metrics: Dict[str, float] = Field(description="Quantitative risk metrics")


# Fund Data Responses
class FundMetrics(BaseModel):
    """Fund performance metrics."""
    annual_return: float = Field(description="Annualized return")
    volatility: float = Field(description="Annual volatility")
    sharpe_ratio: float = Field(description="Sharpe ratio")
    max_drawdown: float = Field(description="Maximum drawdown")
    var_95: float = Field(description="95% Value at Risk")
    skewness: float = Field(description="Return distribution skewness")
    kurtosis: float = Field(description="Return distribution kurtosis")
    calmar_ratio: float = Field(description="Calmar ratio")


class FundSummary(BaseModel):
    """Fund summary information."""
    fund_code: str = Field(description="Fund identifier")
    name: str = Field(description="Fund name")
    category: str = Field(description="Fund category")
    region: str = Field(description="Geographic focus")
    currency: str = Field(description="Base currency")
    ytd_return: float = Field(description="Year-to-date return")
    volatility: float = Field(description="Recent volatility")
    data_quality: DataQuality = Field(description="Data quality assessment")


class FundDetailsResponse(BaseResponse):
    """Detailed fund information response."""
    fund_code: str = Field(description="Fund identifier")
    name: str = Field(description="Fund name")
    category: str = Field(description="Fund category")
    region: str = Field(description="Geographic region")
    description: str = Field(description="Fund description")
    current_metrics: FundMetrics = Field(description="Current performance metrics")
    data_availability: Dict[str, Any] = Field(description="Data availability information")
    last_updated: datetime = Field(description="Last data update timestamp")


class HistoricalDataPoint(BaseModel):
    """Single historical data point."""
    date: date = Field(description="Data date")
    close: float = Field(description="Closing price")
    daily_return: float = Field(description="Daily return")
    volatility: Optional[float] = Field(default=None, description="Rolling volatility")
    volume: Optional[int] = Field(default=None, description="Trading volume")


class FundHistoricalResponse(BaseResponse):
    """Fund historical data response."""
    fund_code: str = Field(description="Fund identifier")
    period: Dict[str, Union[date, int]] = Field(description="Data period information")
    data: List[HistoricalDataPoint] = Field(description="Historical data points")
    summary_metrics: FundMetrics = Field(description="Period summary metrics")
    data_quality: DataQuality = Field(description="Data quality assessment")
    technical_indicators: Optional[Dict[str, List[float]]] = Field(
        default=None, 
        description="Technical indicators if requested"
    )


class FundListResponse(BaseResponse):
    """Fund list response."""
    funds: List[FundSummary] = Field(description="List of funds")
    total_count: int = Field(description="Total number of funds")
    categories: Dict[str, int] = Field(description="Fund count by category")
    regions: Dict[str, int] = Field(description="Fund count by region")
    filters_applied: Dict[str, Any] = Field(description="Applied filters")


class FundComparisonResponse(BaseResponse):
    """Fund comparison response."""
    funds: Dict[str, FundMetrics] = Field(description="Fund performance metrics")
    comparison_period: str = Field(description="Analysis period")
    correlations: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None, 
        description="Correlation matrix if requested"
    )
    rankings: Dict[str, List[str]] = Field(description="Performance rankings by metric")
    summary: Dict[str, Any] = Field(description="Comparison summary")


# Swedish Economic Data Responses
class KeyEconomicIndicators(BaseModel):
    """Key Swedish economic indicators."""
    repo_rate: float = Field(description="Riksbank repo rate")
    inflation_cpi: float = Field(description="CPI inflation rate")
    gdp_growth: float = Field(description="GDP growth rate")
    unemployment_rate: float = Field(description="Unemployment rate")
    house_price_change: float = Field(description="House price change")
    manufacturing_pmi: float = Field(description="Manufacturing PMI")
    consumer_confidence: float = Field(description="Consumer confidence index")


class EconomicTrends(BaseModel):
    """Economic trend analysis."""
    interest_rate_direction: str = Field(description="Interest rate trend direction")
    inflation_trend: str = Field(description="Inflation trend")
    growth_momentum: str = Field(description="Economic growth momentum")
    housing_market_trend: str = Field(description="Housing market trend")
    labor_market_trend: str = Field(description="Labor market trend")


class SwedishEconomicOverviewResponse(BaseResponse):
    """Swedish economic overview response."""
    economic_phase: str = Field(description="Current economic cycle phase")
    key_indicators: KeyEconomicIndicators = Field(description="Key economic indicators")
    trends: EconomicTrends = Field(description="Economic trends")
    investment_implications: List[str] = Field(description="Investment implications")
    last_updated: datetime = Field(description="Data last updated timestamp")
    data_sources: List[str] = Field(description="Data sources used")


class InterestRateDataPoint(BaseModel):
    """Interest rate data point."""
    date: date = Field(description="Data date")
    repo_rate: float = Field(description="Repo rate")
    government_bond_10y: Optional[float] = Field(default=None, description="10-year government bond yield")
    mortgage_rate: Optional[float] = Field(default=None, description="Average mortgage rate")


class InterestRateResponse(BaseResponse):
    """Interest rate data response."""
    current_repo_rate: float = Field(description="Current repo rate")
    trend: str = Field(description="Rate trend direction")
    next_decision_date: date = Field(description="Next monetary policy decision date")
    historical_data: Optional[List[InterestRateDataPoint]] = Field(
        default=None, 
        description="Historical data if requested"
    )
    rate_expectations: Dict[str, float] = Field(description="Market rate expectations")


class InflationComponents(BaseModel):
    """Inflation component breakdown."""
    core_inflation: float = Field(description="Core inflation rate")
    energy_inflation: float = Field(description="Energy price inflation")
    food_inflation: float = Field(description="Food price inflation")
    housing_inflation: float = Field(description="Housing cost inflation")
    services_inflation: float = Field(description="Services inflation")


class InflationDataResponse(BaseResponse):
    """Inflation data response."""
    current_cpi: float = Field(description="Current CPI inflation")
    target: float = Field(description="Inflation target")
    trend: str = Field(description="Inflation trend")
    components: InflationComponents = Field(description="Inflation components")
    expectations: Dict[str, float] = Field(description="Inflation expectations")


class HousingPriceChanges(BaseModel):
    """Housing price change breakdown."""
    monthly: float = Field(description="Monthly change")
    quarterly: float = Field(description="Quarterly change")
    annual: float = Field(description="Annual change")


class HousingDataResponse(BaseResponse):
    """Housing market data response."""
    price_index: float = Field(description="House price index")
    changes: HousingPriceChanges = Field(description="Price changes")
    regional_data: Dict[str, float] = Field(description="Regional price changes")
    market_sentiment: str = Field(description="Housing market sentiment")
    affordability_index: float = Field(description="Housing affordability index")


# Market Analysis Responses
class MarketSentimentFactor(BaseModel):
    """Market sentiment factor."""
    factor: str = Field(description="Sentiment factor name")
    impact: str = Field(description="Impact direction (positive/negative/neutral)")
    weight: float = Field(ge=0, le=1, description="Factor weight in overall sentiment")
    description: str = Field(description="Factor description")


class MarketSentimentResponse(BaseResponse):
    """Market sentiment analysis response."""
    overall_sentiment: MarketSentiment = Field(description="Overall market sentiment")
    confidence_score: float = Field(ge=0, le=100, description="Sentiment confidence score")
    factors: List[MarketSentimentFactor] = Field(description="Sentiment driving factors")
    trend_direction: str = Field(description="Sentiment trend direction")
    volatility_level: str = Field(description="Market volatility level")


class DailyInsightsResponse(BaseResponse):
    """Daily market insights response."""
    date: date = Field(description="Insights date")
    market_summary: str = Field(description="Market summary")
    key_developments: List[str] = Field(description="Key market developments")
    recommendations: List[str] = Field(description="Daily recommendations")
    economic_context: str = Field(description="Economic context")
    sentiment_analysis: MarketSentimentResponse = Field(description="Market sentiment")


# AI Analysis Responses
class InvestmentPlan(BaseModel):
    """Investment implementation plan."""
    fund_amounts: Dict[str, float] = Field(description="Fund allocation amounts in SEK")
    total_amount: float = Field(description="Total investment amount")
    estimated_annual_fees: float = Field(description="Estimated annual fees")
    rebalancing_schedule: Dict[str, Any] = Field(description="Rebalancing schedule")
    action_items: List[str] = Field(description="Implementation action items")
    timeline: str = Field(description="Implementation timeline")


class InvestmentRecommendationResponse(BaseResponse):
    """AI investment recommendation response."""
    recommended_allocation: Dict[str, float] = Field(description="Recommended allocation")
    investment_plan: InvestmentPlan = Field(description="Implementation plan")
    comprehensive_report: str = Field(description="Detailed AI-generated report")
    confidence_rating: float = Field(ge=0, le=1, description="Recommendation confidence")
    risk_assessment: Dict[str, Any] = Field(description="Risk assessment")
    expected_outcomes: Dict[str, Any] = Field(description="Expected investment outcomes")


# System Responses
class ComponentStatus(BaseModel):
    """Individual component status."""
    status: str = Field(description="Component status")
    last_check: datetime = Field(description="Last status check timestamp")
    response_time_ms: Optional[float] = Field(default=None, description="Response time in milliseconds")
    error_message: Optional[str] = Field(default=None, description="Error message if unhealthy")


class HealthCheckResponse(BaseResponse):
    """System health check response."""
    status: str = Field(description="Overall system status")
    version: str = Field(description="API version")
    uptime_seconds: float = Field(description="System uptime in seconds")
    components: Dict[str, ComponentStatus] = Field(description="Component status details")


class PerformanceMetrics(BaseModel):
    """System performance metrics."""
    avg_response_time_ms: float = Field(description="Average response time")
    requests_per_minute: float = Field(description="Current requests per minute")
    error_rate: float = Field(description="Error rate percentage")
    active_connections: int = Field(description="Active connections")


class DataFreshness(BaseModel):
    """Data freshness information."""
    swedish_economic_data: datetime = Field(description="Swedish economic data last updated")
    fund_price_data: datetime = Field(description="Fund price data last updated")
    historical_cache: datetime = Field(description="Historical cache last updated")
    ai_model_data: datetime = Field(description="AI model data last updated")


class SystemStatusResponse(BaseResponse):
    """System status response."""
    api_status: str = Field(description="API status")
    data_freshness: DataFreshness = Field(description="Data freshness information")
    performance_metrics: PerformanceMetrics = Field(description="Performance metrics")
    resource_usage: Dict[str, Any] = Field(description="Resource usage information")
    maintenance_windows: List[Dict[str, Any]] = Field(description="Scheduled maintenance windows")


# Batch Response Models
class BatchResponse(BaseResponse):
    """Batch operation response."""
    total_requests: int = Field(description="Total number of requests processed")
    successful_requests: int = Field(description="Number of successful requests")
    failed_requests: int = Field(description="Number of failed requests")
    results: List[Dict[str, Any]] = Field(description="Individual request results")
    execution_time_seconds: float = Field(description="Total execution time")


class PaginatedResponse(BaseResponse):
    """Paginated response model."""
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Items per page")
    total_items: int = Field(description="Total number of items")
    total_pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Whether there are more pages")
    has_previous: bool = Field(description="Whether there are previous pages")