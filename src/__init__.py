"""
Investment MCP System - Unified Implementation

A comprehensive Swedish investment analysis and portfolio optimization system
providing AI-enhanced recommendations for the approved fund universe.

This is the streamlined, consolidated version of the Investment MCP system
that eliminates redundancy while maintaining all core functionality.
"""

__version__ = "4.0.0"
__author__ = "Investment MCP Team"
__description__ = "Unified Swedish Investment Analysis and Portfolio Optimization System"

# Core components
from .core import (
    get_settings,
    TRADEABLE_FUNDS,
    FUND_UNIVERSE,
    get_approved_funds,
    validate_fund_allocation
)

# Data collection
from .collectors import (
    MarketDataCollector,
    SwedishDataCollector,
    BaseDataCollector
)

# Analysis engines
from .analysis import (
    RiskAnalyzer,
    TechnicalAnalyzer,
    PortfolioOptimizer
)

# AI services
from .ai import (
    AIProviderManager,
    AIPortfolioAdvisor
)

# MCP server
from .mcp_tools import (
    InvestmentMCPServer,
    create_investment_mcp_server
)

# Utilities
from .utils import (
    setup_logging,
    get_logger,
    format_currency,
    format_percentage,
    get_cache
)

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__description__',
    
    # Core
    'get_settings',
    'TRADEABLE_FUNDS',
    'FUND_UNIVERSE', 
    'get_approved_funds',
    'validate_fund_allocation',
    
    # Data collection
    'MarketDataCollector',
    'SwedishDataCollector',
    'BaseDataCollector',
    
    # Analysis
    'RiskAnalyzer',
    'TechnicalAnalyzer',
    'PortfolioOptimizer',
    
    # AI services
    'AIProviderManager',
    'AIPortfolioAdvisor',
    
    # MCP server
    'InvestmentMCPServer',
    'create_investment_mcp_server',
    
    # Utilities
    'setup_logging',
    'get_logger',
    'format_currency',
    'format_percentage',
    'get_cache'
]


def get_system_info() -> dict:
    """Get comprehensive system information."""
    
    settings = get_settings()
    
    return {
        "system": {
            "name": "Investment MCP System",
            "version": __version__,
            "description": __description__,
            "environment": settings.ENVIRONMENT
        },
        "capabilities": {
            "fund_universe_size": len(TRADEABLE_FUNDS),
            "approved_funds": list(TRADEABLE_FUNDS.keys()),
            "data_collection": [
                "Swedish fund market data (20+ years)",
                "Riksbanken economic indicators",
                "SCB statistics"
            ],
            "analysis_engines": [
                "Technical analysis (RSI, MACD, Bollinger Bands)",
                "Risk analysis (VaR, Sharpe ratio, max drawdown)",
                "Portfolio optimization (mean-variance, risk parity)"
            ],
            "ai_services": [
                "Portfolio recommendations",
                "Fund selection explanations", 
                "Market sentiment analysis"
            ],
            "mcp_tools": [
                "Data collection",
                "Risk analysis", 
                "Technical analysis",
                "Portfolio optimization",
                "AI recommendations"
            ]
        },
        "configuration": {
            "ai_provider": settings.AI_PROVIDER,
            "historical_data_years": settings.HISTORICAL_DATA_YEARS,
            "database_url": settings.DATABASE_URL.replace(settings.SECRET_KEY, "***") if hasattr(settings, 'SECRET_KEY') else settings.DATABASE_URL,
            "debug_mode": settings.DEBUG
        }
    }


def quick_start_example():
    """Provide quick start example code."""
    
    example_code = '''
# Investment MCP System - Quick Start Example

from investment_mcp import (
    MarketDataCollector,
    PortfolioOptimizer, 
    AIPortfolioAdvisor,
    create_investment_mcp_server,
    get_approved_funds
)

# 1. Get approved fund universe
approved_funds = get_approved_funds()
print(f"Approved funds: {len(approved_funds)}")

# 2. Collect market data
async def collect_data():
    collector = MarketDataCollector()
    data = await collector.collect_historical_data(years=5)
    return data

# 3. Optimize portfolio
def optimize_portfolio():
    optimizer = PortfolioOptimizer()
    result = optimizer.mean_variance_optimization(
        returns_matrix=your_returns_data,
        risk_tolerance="medium"
    )
    return result

# 4. Get AI recommendation
async def get_ai_recommendation():
    advisor = AIPortfolioAdvisor()
    recommendation = await advisor.generate_portfolio_recommendation(
        risk_tolerance="medium",
        investment_horizon="long", 
        investment_amount=100000
    )
    return recommendation

# 5. Start MCP server
async def start_server():
    server = create_investment_mcp_server()
    await server.run()
'''
    
    return example_code


# Initialize logging when package is imported
try:
    setup_logging()
    logger = get_logger(__name__)
    logger.info(f"Investment MCP System v{__version__} initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize logging: {e}")

print(f"Investment MCP System v{__version__} - Unified Swedish Investment Analysis")
print(f"Approved funds: {len(TRADEABLE_FUNDS)} | Environment: {get_settings().ENVIRONMENT}")
print("Ready for investment analysis and portfolio optimization.")