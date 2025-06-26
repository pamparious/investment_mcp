"""
Core package for Investment MCP System.

This package contains the foundational components including configuration,
database models, and fund universe definitions.
"""

from .config import (
    InvestmentMCPSettings,
    get_settings,
    get_approved_funds,
    get_fund_info,
    validate_fund_allocation,
    get_fund_universe,
    get_environment_info,
    FUND_UNIVERSE,
    TRADEABLE_FUNDS
)

from .database import (
    Base,
    MarketData,
    EconomicData,
    AnalysisResult,
    Portfolio,
    PortfolioHolding,
    OptimizationResult,
    DataCollectionLog,
    AIAnalysisLog,
    SwedishEconomicIndicator,
    validate_fund_allocation as db_validate_fund_allocation
)

__all__ = [
    # Configuration
    'InvestmentMCPSettings',
    'get_settings',
    'get_approved_funds',
    'get_fund_info',
    'validate_fund_allocation',
    'get_fund_universe',
    'get_environment_info',
    'FUND_UNIVERSE',
    'TRADEABLE_FUNDS',
    
    # Database models
    'Base',
    'MarketData',
    'EconomicData',
    'AnalysisResult',
    'Portfolio',
    'PortfolioHolding',
    'OptimizationResult',
    'DataCollectionLog',
    'AIAnalysisLog',
    'SwedishEconomicIndicator',
    'db_validate_fund_allocation'
]