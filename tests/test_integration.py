"""
Integration tests for the unified Investment MCP System.

This module provides comprehensive integration tests to ensure all components
work together correctly after the cleanup and consolidation.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src import (
    get_settings,
    TRADEABLE_FUNDS,
    get_approved_funds,
    validate_fund_allocation,
    MarketDataCollector,
    SwedishDataCollector,
    RiskAnalyzer,
    TechnicalAnalyzer,
    PortfolioOptimizer,
    AIPortfolioAdvisor,
    create_investment_mcp_server
)


class TestSystemIntegration:
    """Test system-wide integration and functionality."""
    
    def test_system_initialization(self):
        """Test that the system initializes correctly."""
        settings = get_settings()
        assert settings is not None
        assert settings.APP_NAME == "Investment MCP System"
        assert settings.VERSION == "4.0.0"
    
    def test_fund_universe_consistency(self):
        """Test that fund universe is consistent across components."""
        approved_funds = get_approved_funds()
        assert len(approved_funds) == 12  # Exact number as specified
        
        # Check that all funds are in TRADEABLE_FUNDS
        for fund_id in approved_funds:
            assert fund_id in TRADEABLE_FUNDS
            fund_info = TRADEABLE_FUNDS[fund_id]
            assert "name" in fund_info
            assert "category" in fund_info
            assert "risk_level" in fund_info
    
    def test_fund_allocation_validation(self):
        """Test fund allocation validation."""
        # Valid allocation
        valid_allocation = {
            "DNB_GLOBAL_INDEKS_S": 0.4,
            "AVANZA_USA": 0.3,
            "PLUS_ALLABOLAG_SVERIGE_INDEX": 0.3
        }
        
        result = validate_fund_allocation(valid_allocation)
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        
        # Invalid allocation (doesn't sum to 1)
        invalid_allocation = {
            "DNB_GLOBAL_INDEKS_S": 0.5,
            "AVANZA_USA": 0.3
        }
        
        result = validate_fund_allocation(invalid_allocation)
        assert result["valid"] is False
        assert len(result["errors"]) > 0


class TestDataCollectionIntegration:
    """Test data collection component integration."""
    
    @pytest.mark.asyncio
    async def test_market_data_collector_initialization(self):
        """Test market data collector initializes correctly."""
        collector = MarketDataCollector()
        assert collector.settings is not None
        assert collector.timeout > 0
        
        # Test fund ticker mapping
        assert len(collector.FUND_TICKERS) == 12
        for fund_id in get_approved_funds():
            assert fund_id in collector.FUND_TICKERS
    
    @pytest.mark.asyncio
    async def test_swedish_data_collector_initialization(self):
        """Test Swedish data collector initializes correctly."""
        collector = SwedishDataCollector()
        assert collector.settings is not None
        assert collector.riksbank_base_url is not None
        assert collector.scb_base_url is not None
        
        # Test series configuration
        assert len(collector.RIKSBANK_SERIES) > 0
        assert len(collector.SCB_INDICATORS) > 0
    
    @pytest.mark.asyncio
    async def test_data_collection_with_mock_data(self):
        """Test data collection with mocked responses."""
        collector = MarketDataCollector()
        
        # Create mock data
        mock_data = {
            "DNB_GLOBAL_INDEKS_S": pd.DataFrame({
                "date": pd.date_range("2023-01-01", "2023-12-31", freq="D"),
                "open": np.random.randn(365) + 100,
                "high": np.random.randn(365) + 105,
                "low": np.random.randn(365) + 95,
                "close": np.random.randn(365) + 100,
                "volume": np.random.randint(1000, 10000, 365)
            })
        }
        
        # Test statistics calculation
        statistics = collector.get_fund_statistics(mock_data)
        assert "DNB_GLOBAL_INDEKS_S" in statistics
        assert statistics["DNB_GLOBAL_INDEKS_S"]["status"] == "success"


class TestAnalysisIntegration:
    """Test analysis component integration."""
    
    def test_risk_analyzer_initialization(self):
        """Test risk analyzer initializes correctly."""
        analyzer = RiskAnalyzer()
        assert analyzer.logger is not None
    
    def test_technical_analyzer_initialization(self):
        """Test technical analyzer initializes correctly."""
        analyzer = TechnicalAnalyzer()
        assert analyzer.logger is not None
    
    def test_portfolio_optimizer_initialization(self):
        """Test portfolio optimizer initializes correctly."""
        optimizer = PortfolioOptimizer()
        assert optimizer.approved_funds == get_approved_funds()
    
    def test_risk_analysis_with_mock_data(self):
        """Test risk analysis with mock returns data."""
        analyzer = RiskAnalyzer()
        
        # Create mock returns
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns for 1 year
        
        metrics = analyzer.calculate_basic_risk_metrics(returns)
        
        # Check that all expected metrics are present
        expected_metrics = [
            "mean_return", "std_return", "annualized_return", 
            "annualized_volatility", "sharpe_ratio", "max_drawdown"
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_portfolio_optimization_with_mock_data(self):
        """Test portfolio optimization with mock data."""
        optimizer = PortfolioOptimizer()
        
        # Create mock returns matrix
        np.random.seed(42)
        fund_ids = get_approved_funds()[:6]  # Use first 6 funds
        dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
        
        returns_data = {}
        for i, fund_id in enumerate(fund_ids):
            returns_data[fund_id] = np.random.normal(0.0008, 0.02, len(dates))
        
        returns_matrix = pd.DataFrame(returns_data, index=dates)
        
        # Test mean-variance optimization
        result = optimizer.mean_variance_optimization(
            returns_matrix, 
            risk_tolerance="medium"
        )
        
        assert result["success"] is True
        assert "fund_allocation" in result
        assert "portfolio_metrics" in result
        
        # Validate allocation
        allocation = result["fund_allocation"]
        validation = validate_fund_allocation(allocation)
        assert validation["valid"] is True


class TestAIIntegration:
    """Test AI component integration."""
    
    @pytest.mark.asyncio
    async def test_ai_advisor_initialization(self):
        """Test AI advisor initializes correctly."""
        advisor = AIPortfolioAdvisor()
        assert advisor.ai_manager is not None
        assert advisor.portfolio_optimizer is not None
        assert advisor.risk_analyzer is not None
    
    @pytest.mark.asyncio
    async def test_ai_recommendation_fallback(self):
        """Test AI recommendation with fallback when AI fails."""
        advisor = AIPortfolioAdvisor()
        
        # Mock AI failure
        with patch.object(advisor.ai_manager, 'generate_response') as mock_ai:
            mock_ai.return_value = {"success": False, "error": "AI unavailable"}
            
            recommendation = await advisor.generate_portfolio_recommendation(
                risk_tolerance="medium",
                investment_horizon="long",
                investment_amount=100000
            )
            
            assert "portfolio_allocation" in recommendation
            assert "ai_insights" in recommendation
            # Should still provide a recommendation even with AI failure


class TestMCPServerIntegration:
    """Test MCP server integration."""
    
    def test_mcp_server_creation(self):
        """Test MCP server can be created."""
        server = create_investment_mcp_server()
        assert server is not None
        assert hasattr(server, 'server')
        assert hasattr(server, 'market_collector')
        assert hasattr(server, 'portfolio_optimizer')
        assert hasattr(server, 'ai_advisor')
    
    def test_mcp_server_components(self):
        """Test MCP server has all required components."""
        server = create_investment_mcp_server()
        
        # Check all major components are initialized
        assert server.market_collector is not None
        assert server.swedish_collector is not None
        assert server.risk_analyzer is not None
        assert server.technical_analyzer is not None
        assert server.portfolio_optimizer is not None
        assert server.ai_advisor is not None


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_portfolio_analysis_workflow(self):
        """Test complete workflow from data collection to recommendation."""
        
        # Step 1: Initialize components
        market_collector = MarketDataCollector()
        risk_analyzer = RiskAnalyzer()
        portfolio_optimizer = PortfolioOptimizer()
        
        # Step 2: Mock data collection (in real scenario would collect actual data)
        np.random.seed(42)
        fund_ids = get_approved_funds()[:5]
        dates = pd.date_range("2022-01-01", "2023-12-31", freq="D")
        
        # Create mock price data
        mock_data = {}
        for fund_id in fund_ids:
            prices = 100 * np.cumprod(1 + np.random.normal(0.0008, 0.02, len(dates)))
            mock_data[fund_id] = pd.DataFrame({
                "date": dates,
                "close": prices
            })
        
        # Step 3: Calculate returns
        returns_matrix = pd.DataFrame()
        for fund_id, data in mock_data.items():
            returns = data["close"].pct_change().dropna()
            returns_matrix[fund_id] = returns
        
        # Step 4: Perform risk analysis
        portfolio_allocation = {fund_id: 1.0/len(fund_ids) for fund_id in fund_ids}
        risk_metrics = risk_analyzer.calculate_portfolio_risk(returns_matrix, portfolio_allocation)
        
        assert "annualized_return" in risk_metrics
        assert "annualized_volatility" in risk_metrics
        assert "sharpe_ratio" in risk_metrics
        
        # Step 5: Optimize portfolio
        optimization_result = portfolio_optimizer.mean_variance_optimization(
            returns_matrix, risk_tolerance="medium"
        )
        
        assert optimization_result["success"] is True
        optimized_allocation = optimization_result["fund_allocation"]
        
        # Validate optimized allocation
        validation = validate_fund_allocation(optimized_allocation)
        assert validation["valid"] is True
        
        # Step 6: Verify allocation uses only approved funds
        for fund_id in optimized_allocation.keys():
            assert fund_id in get_approved_funds()
    
    def test_model_portfolio_consistency(self):
        """Test that model portfolios are consistent and valid."""
        optimizer = PortfolioOptimizer()
        model_portfolios = optimizer.create_model_portfolios()
        
        # Check all model portfolios
        expected_portfolios = ["conservative", "balanced", "growth", "aggressive"]
        
        for portfolio_name in expected_portfolios:
            assert portfolio_name in model_portfolios
            
            portfolio = model_portfolios[portfolio_name]
            assert "fund_allocation" in portfolio
            assert "risk_tolerance" in portfolio
            
            # Validate allocation
            allocation = portfolio["fund_allocation"]
            validation = validate_fund_allocation(allocation)
            assert validation["valid"] is True, f"Invalid allocation in {portfolio_name} portfolio"
            
            # Check that all funds are approved
            for fund_id in allocation.keys():
                assert fund_id in get_approved_funds()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_data_handling(self):
        """Test handling of empty data scenarios."""
        risk_analyzer = RiskAnalyzer()
        
        # Test with empty returns
        empty_returns = pd.Series()
        metrics = risk_analyzer.calculate_basic_risk_metrics(empty_returns)
        
        # Should return empty metrics without crashing
        assert isinstance(metrics, dict)
        assert metrics.get("annualized_return", 0) == 0
    
    def test_invalid_fund_allocation(self):
        """Test handling of invalid fund allocations."""
        
        # Test with non-approved fund
        invalid_allocation = {
            "INVALID_FUND": 0.5,
            "DNB_GLOBAL_INDEKS_S": 0.5
        }
        
        result = validate_fund_allocation(invalid_allocation)
        assert result["valid"] is False
        assert any("not in approved universe" in error for error in result["errors"])
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        settings = get_settings()
        
        # Test that required settings are present
        assert hasattr(settings, 'APP_NAME')
        assert hasattr(settings, 'VERSION')
        assert hasattr(settings, 'HISTORICAL_DATA_YEARS')
        assert hasattr(settings, 'AI_PROVIDER')
        
        # Test that fund universe is properly configured
        assert len(TRADEABLE_FUNDS) == 12
        assert len(get_approved_funds()) == 12


@pytest.fixture
def sample_returns_data():
    """Fixture providing sample returns data for testing."""
    np.random.seed(42)
    fund_ids = get_approved_funds()[:5]
    dates = pd.date_range("2022-01-01", "2023-12-31", freq="D")
    
    returns_data = {}
    for i, fund_id in enumerate(fund_ids):
        # Different risk/return profiles
        mean_return = 0.0005 + i * 0.0001
        volatility = 0.015 + i * 0.002
        returns_data[fund_id] = np.random.normal(mean_return, volatility, len(dates))
    
    return pd.DataFrame(returns_data, index=dates)


@pytest.fixture
def sample_market_data():
    """Fixture providing sample market data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    
    # Generate realistic OHLCV data
    initial_price = 100
    returns = np.random.normal(0.0008, 0.02, len(dates))
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = {
        "date": dates,
        "close_price": prices,
        "open_price": [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        "high_price": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        "low_price": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        "volume": np.random.randint(10000, 100000, len(dates))
    }
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])