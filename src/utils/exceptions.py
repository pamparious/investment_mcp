"""Custom exceptions for the Investment MCP System."""

from typing import Optional, Any, Dict


class InvestmentMCPError(Exception):
    """Base exception for Investment MCP System."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(InvestmentMCPError):
    """Raised when there's a configuration error."""
    pass


class DataCollectionError(InvestmentMCPError):
    """Raised when data collection fails."""
    pass


class DataQualityError(InvestmentMCPError):
    """Raised when data quality is insufficient."""
    pass


class AnalysisError(InvestmentMCPError):
    """Raised when analysis computation fails."""
    pass


class OptimizationError(InvestmentMCPError):
    """Raised when portfolio optimization fails."""
    pass


class ValidationError(InvestmentMCPError):
    """Raised when data validation fails."""
    pass


class APIError(InvestmentMCPError):
    """Raised when API operations fail."""
    pass


class DatabaseError(InvestmentMCPError):
    """Raised when database operations fail."""
    pass


class ExternalServiceError(InvestmentMCPError):
    """Raised when external service calls fail."""
    pass


class InsufficientDataError(DataCollectionError):
    """Raised when there's insufficient data for analysis."""
    pass


class FundNotFoundError(InvestmentMCPError):
    """Raised when a requested fund is not found."""
    pass


class InvalidAllocationError(ValidationError):
    """Raised when portfolio allocation is invalid."""
    pass


class RiskCalculationError(AnalysisError):
    """Raised when risk calculations fail."""
    pass


class BacktestError(AnalysisError):
    """Raised when backtesting fails."""
    pass


class MCPAgentError(InvestmentMCPError):
    """Raised when MCP agent operations fail."""
    pass