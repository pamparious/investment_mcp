"""Custom exceptions for Investment MCP API."""

from typing import Any, Dict, List, Optional


class APIException(Exception):
    """Base exception for API errors."""
    
    def __init__(
        self,
        detail: str,
        status_code: int = 500,
        error_type: str = "api_error",
        headers: Optional[Dict[str, str]] = None
    ):
        self.detail = detail
        self.status_code = status_code
        self.error_type = error_type
        self.headers = headers or {}
        super().__init__(detail)


class ValidationException(APIException):
    """Exception for request validation errors."""
    
    def __init__(
        self,
        detail: str = "Validation failed",
        errors: Optional[List[Dict[str, Any]]] = None,
        field: Optional[str] = None
    ):
        super().__init__(detail, status_code=400, error_type="validation_error")
        self.errors = errors or []
        self.field = field


class AuthenticationException(APIException):
    """Exception for authentication failures."""
    
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            detail, 
            status_code=401, 
            error_type="authentication_error",
            headers={"WWW-Authenticate": "Bearer"}
        )


class AuthorizationException(APIException):
    """Exception for authorization failures."""
    
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(detail, status_code=403, error_type="authorization_error")


class ResourceNotFoundException(APIException):
    """Exception for resource not found errors."""
    
    def __init__(self, resource_type: str, resource_id: str):
        detail = f"{resource_type} with ID '{resource_id}' not found"
        super().__init__(detail, status_code=404, error_type="not_found_error")
        self.resource_type = resource_type
        self.resource_id = resource_id


class RateLimitException(APIException):
    """Exception for rate limit exceeded."""
    
    def __init__(
        self, 
        detail: str = "Rate limit exceeded",
        retry_after: int = 60,
        limit_type: str = "general"
    ):
        headers = {"Retry-After": str(retry_after)}
        super().__init__(
            detail, 
            status_code=429, 
            error_type="rate_limit_error",
            headers=headers
        )
        self.retry_after = retry_after
        self.limit_type = limit_type


class DataQualityException(APIException):
    """Exception for data quality issues."""
    
    def __init__(
        self, 
        detail: str = "Data quality insufficient for analysis",
        data_source: Optional[str] = None,
        quality_score: Optional[float] = None
    ):
        super().__init__(detail, status_code=422, error_type="data_quality_error")
        self.data_source = data_source
        self.quality_score = quality_score


class ExternalServiceException(APIException):
    """Exception for external service failures."""
    
    def __init__(
        self,
        detail: str = "External service unavailable",
        service_name: Optional[str] = None,
        original_error: Optional[str] = None
    ):
        super().__init__(detail, status_code=503, error_type="external_service_error")
        self.service_name = service_name
        self.original_error = original_error


class AIServiceException(APIException):
    """Exception for AI service failures."""
    
    def __init__(
        self,
        detail: str = "AI service error",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        original_error: Optional[str] = None
    ):
        super().__init__(detail, status_code=502, error_type="ai_service_error")
        self.provider = provider
        self.model = model
        self.original_error = original_error


class AnalysisException(APIException):
    """Exception for analysis failures."""
    
    def __init__(
        self,
        detail: str = "Analysis failed",
        analysis_type: Optional[str] = None,
        reason: Optional[str] = None
    ):
        super().__init__(detail, status_code=422, error_type="analysis_error")
        self.analysis_type = analysis_type
        self.reason = reason


class ConfigurationException(APIException):
    """Exception for configuration issues."""
    
    def __init__(
        self,
        detail: str = "Configuration error",
        config_key: Optional[str] = None
    ):
        super().__init__(detail, status_code=500, error_type="configuration_error")
        self.config_key = config_key


class DatabaseException(APIException):
    """Exception for database operation failures."""
    
    def __init__(
        self,
        detail: str = "Database operation failed",
        operation: Optional[str] = None,
        table: Optional[str] = None
    ):
        super().__init__(detail, status_code=500, error_type="database_error")
        self.operation = operation
        self.table = table


class CacheException(APIException):
    """Exception for cache operation failures."""
    
    def __init__(
        self,
        detail: str = "Cache operation failed",
        operation: Optional[str] = None,
        key: Optional[str] = None
    ):
        super().__init__(detail, status_code=500, error_type="cache_error")
        self.operation = operation
        self.key = key


class BusinessLogicException(APIException):
    """Exception for business logic violations."""
    
    def __init__(
        self,
        detail: str = "Business logic violation",
        rule: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(detail, status_code=422, error_type="business_logic_error")
        self.rule = rule
        self.context = context or {}


class FundNotFoundException(ResourceNotFoundException):
    """Exception for fund not found errors."""
    
    def __init__(self, fund_code: str):
        super().__init__("Fund", fund_code)
        self.fund_code = fund_code


class InvalidAllocationException(ValidationException):
    """Exception for invalid portfolio allocation."""
    
    def __init__(
        self,
        detail: str = "Invalid portfolio allocation",
        allocation: Optional[Dict[str, float]] = None,
        violation: Optional[str] = None
    ):
        super().__init__(detail)
        self.allocation = allocation
        self.violation = violation


class InsufficientDataException(DataQualityException):
    """Exception for insufficient data for analysis."""
    
    def __init__(
        self,
        detail: str = "Insufficient data for analysis",
        required_period: Optional[str] = None,
        available_period: Optional[str] = None,
        fund_code: Optional[str] = None
    ):
        super().__init__(detail)
        self.required_period = required_period
        self.available_period = available_period
        self.fund_code = fund_code


class MarketDataException(ExternalServiceException):
    """Exception for market data service failures."""
    
    def __init__(
        self,
        detail: str = "Market data service unavailable",
        symbol: Optional[str] = None,
        provider: Optional[str] = None
    ):
        super().__init__(detail, service_name=provider)
        self.symbol = symbol
        self.provider = provider


class EconomicDataException(ExternalServiceException):
    """Exception for economic data service failures."""
    
    def __init__(
        self,
        detail: str = "Economic data service unavailable",
        indicator: Optional[str] = None,
        source: Optional[str] = None
    ):
        super().__init__(detail, service_name=source)
        self.indicator = indicator
        self.source = source


class PortfolioOptimizationException(AnalysisException):
    """Exception for portfolio optimization failures."""
    
    def __init__(
        self,
        detail: str = "Portfolio optimization failed",
        optimization_method: Optional[str] = None,
        convergence_issue: bool = False
    ):
        super().__init__(detail, analysis_type="portfolio_optimization")
        self.optimization_method = optimization_method
        self.convergence_issue = convergence_issue


class StressTestException(AnalysisException):
    """Exception for stress testing failures."""
    
    def __init__(
        self,
        detail: str = "Stress test failed",
        scenario: Optional[str] = None,
        data_issue: bool = False
    ):
        super().__init__(detail, analysis_type="stress_test")
        self.scenario = scenario
        self.data_issue = data_issue


class RiskCalculationException(AnalysisException):
    """Exception for risk calculation failures."""
    
    def __init__(
        self,
        detail: str = "Risk calculation failed",
        metric: Optional[str] = None,
        mathematical_error: bool = False
    ):
        super().__init__(detail, analysis_type="risk_calculation")
        self.metric = metric
        self.mathematical_error = mathematical_error


# Exception utility functions
def create_validation_error(
    field: str, 
    value: Any, 
    constraint: str,
    message: Optional[str] = None
) -> ValidationException:
    """Create a validation exception for a specific field."""
    
    default_message = f"Field '{field}' with value '{value}' violates constraint: {constraint}"
    detail = message or default_message
    
    errors = [{
        "field": field,
        "value": value,
        "constraint": constraint,
        "message": detail
    }]
    
    return ValidationException(detail=detail, errors=errors, field=field)


def create_business_rule_error(
    rule_name: str,
    description: str,
    context: Optional[Dict[str, Any]] = None
) -> BusinessLogicException:
    """Create a business logic exception for a specific rule."""
    
    detail = f"Business rule violation: {rule_name} - {description}"
    
    return BusinessLogicException(
        detail=detail,
        rule=rule_name,
        context=context or {}
    )


def create_service_unavailable_error(
    service_name: str,
    operation: str,
    original_error: Optional[Exception] = None
) -> ExternalServiceException:
    """Create an external service exception."""
    
    detail = f"Service '{service_name}' unavailable for operation '{operation}'"
    original_error_str = str(original_error) if original_error else None
    
    return ExternalServiceException(
        detail=detail,
        service_name=service_name,
        original_error=original_error_str
    )


def wrap_database_error(
    original_error: Exception,
    operation: str,
    table: Optional[str] = None
) -> DatabaseException:
    """Wrap database errors in our custom exception."""
    
    detail = f"Database {operation} operation failed"
    if table:
        detail += f" on table '{table}'"
    
    # Add original error message
    detail += f": {str(original_error)}"
    
    return DatabaseException(
        detail=detail,
        operation=operation,
        table=table
    )


def wrap_ai_service_error(
    original_error: Exception,
    provider: str,
    model: Optional[str] = None
) -> AIServiceException:
    """Wrap AI service errors in our custom exception."""
    
    detail = f"AI service error from provider '{provider}'"
    if model:
        detail += f" using model '{model}'"
    
    return AIServiceException(
        detail=detail,
        provider=provider,
        model=model,
        original_error=str(original_error)
    )