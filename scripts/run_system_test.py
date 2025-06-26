#!/usr/bin/env python3
"""
System test script for Investment MCP System.

This script runs comprehensive tests to verify that the unified system
works correctly after cleanup and consolidation.
"""

import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src import (
    get_system_info,
    setup_logging,
    get_logger,
    MarketDataCollector,
    SwedishDataCollector, 
    RiskAnalyzer,
    TechnicalAnalyzer,
    PortfolioOptimizer,
    AIPortfolioAdvisor,
    create_investment_mcp_server,
    get_approved_funds,
    validate_fund_allocation
)


class SystemTester:
    """Comprehensive system tester for Investment MCP."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.test_results = {}
        self.start_time = datetime.utcnow()
    
    def run_all_tests(self):
        """Run all system tests."""
        
        print("=" * 80)
        print("Investment MCP System - Comprehensive Test Suite")
        print("=" * 80)
        
        # Basic system tests
        self.test_system_info()
        self.test_fund_universe()
        self.test_configuration()
        
        # Component initialization tests
        self.test_collectors_initialization()
        self.test_analyzers_initialization() 
        self.test_ai_services_initialization()
        self.test_mcp_server_initialization()
        
        # Integration tests
        asyncio.run(self.test_data_collection_simulation())
        self.test_analysis_pipeline()
        self.test_portfolio_optimization()
        
        # Generate test report
        self.generate_test_report()
    
    def test_system_info(self):
        """Test system information and configuration."""
        
        print("\n1. Testing System Information...")
        
        try:
            system_info = get_system_info()
            
            # Verify basic system info
            assert system_info["system"]["name"] == "Investment MCP System"
            assert system_info["system"]["version"] == "4.0.0"
            assert "capabilities" in system_info
            assert "configuration" in system_info
            
            # Verify capabilities
            capabilities = system_info["capabilities"]
            assert capabilities["fund_universe_size"] == 12
            assert len(capabilities["approved_funds"]) == 12
            assert len(capabilities["data_collection"]) > 0
            assert len(capabilities["analysis_engines"]) > 0
            
            self.test_results["system_info"] = "PASS"
            print("   ✓ System information verified")
            
        except Exception as e:
            self.test_results["system_info"] = f"FAIL: {e}"
            print(f"   ✗ System information test failed: {e}")
    
    def test_fund_universe(self):
        """Test fund universe consistency."""
        
        print("\n2. Testing Fund Universe...")
        
        try:
            approved_funds = get_approved_funds()
            
            # Verify fund count
            assert len(approved_funds) == 12, f"Expected 12 funds, got {len(approved_funds)}"
            
            # Verify all expected funds are present
            expected_funds = [
                "DNB_GLOBAL_INDEKS_S",
                "AVANZA_EMERGING_MARKETS", 
                "STOREBRAND_EUROPA_A_SEK",
                "DNB_NORDEN_INDEKS_S",
                "PLUS_ALLABOLAG_SVERIGE_INDEX",
                "AVANZA_USA",
                "STOREBRAND_JAPAN_A_SEK",
                "HANDELSBANKEN_GLOBAL_SMAB_INDEX",
                "XETRA_GOLD_ETC",
                "VIRTUNE_BITCOIN_PRIME_ETP",
                "XBT_ETHER_ONE",
                "PLUS_FASTIGHETER_SVERIGE_INDEX"
            ]
            
            for fund in expected_funds:
                assert fund in approved_funds, f"Missing fund: {fund}"
            
            # Test allocation validation
            test_allocation = {
                "DNB_GLOBAL_INDEKS_S": 0.4,
                "AVANZA_USA": 0.3,
                "PLUS_ALLABOLAG_SVERIGE_INDEX": 0.3
            }
            
            validation = validate_fund_allocation(test_allocation)
            assert validation["valid"], f"Valid allocation rejected: {validation['errors']}"
            
            self.test_results["fund_universe"] = "PASS"
            print("   ✓ Fund universe verified")
            print(f"   ✓ All {len(approved_funds)} approved funds present")
            print("   ✓ Allocation validation working")
            
        except Exception as e:
            self.test_results["fund_universe"] = f"FAIL: {e}"
            print(f"   ✗ Fund universe test failed: {e}")
    
    def test_configuration(self):
        """Test system configuration."""
        
        print("\n3. Testing Configuration...")
        
        try:
            from src.core.config import get_settings
            
            settings = get_settings()
            
            # Verify required settings
            required_attrs = [
                "APP_NAME", "VERSION", "ENVIRONMENT", "AI_PROVIDER",
                "HISTORICAL_DATA_YEARS", "DATABASE_URL"
            ]
            
            for attr in required_attrs:
                assert hasattr(settings, attr), f"Missing setting: {attr}"
            
            # Verify reasonable values
            assert settings.HISTORICAL_DATA_YEARS >= 1
            assert settings.HISTORICAL_DATA_YEARS <= 25
            assert settings.AI_PROVIDER in ["openai", "anthropic", "ollama"]
            
            self.test_results["configuration"] = "PASS"
            print("   ✓ Configuration settings verified")
            print(f"   ✓ Environment: {settings.ENVIRONMENT}")
            print(f"   ✓ AI Provider: {settings.AI_PROVIDER}")
            
        except Exception as e:
            self.test_results["configuration"] = f"FAIL: {e}"
            print(f"   ✗ Configuration test failed: {e}")
    
    def test_collectors_initialization(self):
        """Test data collector initialization."""
        
        print("\n4. Testing Data Collectors...")
        
        try:
            # Test market data collector
            market_collector = MarketDataCollector()
            assert market_collector.settings is not None
            assert len(market_collector.FUND_TICKERS) == 12
            
            # Test Swedish data collector
            swedish_collector = SwedishDataCollector()
            assert swedish_collector.settings is not None
            assert len(swedish_collector.RIKSBANK_SERIES) > 0
            assert len(swedish_collector.SCB_INDICATORS) > 0
            
            self.test_results["collectors"] = "PASS"
            print("   ✓ Market data collector initialized")
            print("   ✓ Swedish data collector initialized")
            print(f"   ✓ {len(market_collector.FUND_TICKERS)} fund tickers mapped")
            
        except Exception as e:
            self.test_results["collectors"] = f"FAIL: {e}"
            print(f"   ✗ Collectors test failed: {e}")
    
    def test_analyzers_initialization(self):
        """Test analyzer initialization."""
        
        print("\n5. Testing Analyzers...")
        
        try:
            # Test risk analyzer
            risk_analyzer = RiskAnalyzer()
            assert risk_analyzer.logger is not None
            
            # Test technical analyzer
            technical_analyzer = TechnicalAnalyzer()
            assert technical_analyzer.logger is not None
            
            # Test portfolio optimizer
            portfolio_optimizer = PortfolioOptimizer()
            assert portfolio_optimizer.approved_funds is not None
            assert len(portfolio_optimizer.approved_funds) == 12
            
            self.test_results["analyzers"] = "PASS"
            print("   ✓ Risk analyzer initialized")
            print("   ✓ Technical analyzer initialized")
            print("   ✓ Portfolio optimizer initialized")
            
        except Exception as e:
            self.test_results["analyzers"] = f"FAIL: {e}"
            print(f"   ✗ Analyzers test failed: {e}")
    
    def test_ai_services_initialization(self):
        """Test AI services initialization."""
        
        print("\n6. Testing AI Services...")
        
        try:
            # Test AI portfolio advisor
            ai_advisor = AIPortfolioAdvisor()
            assert ai_advisor.ai_manager is not None
            assert ai_advisor.portfolio_optimizer is not None
            assert ai_advisor.risk_analyzer is not None
            
            # Test AI provider manager
            provider_info = ai_advisor.ai_manager.get_provider_info()
            assert isinstance(provider_info, dict)
            
            self.test_results["ai_services"] = "PASS"
            print("   ✓ AI portfolio advisor initialized")
            print(f"   ✓ {len(provider_info)} AI providers configured")
            
        except Exception as e:
            self.test_results["ai_services"] = f"FAIL: {e}"
            print(f"   ✗ AI services test failed: {e}")
    
    def test_mcp_server_initialization(self):
        """Test MCP server initialization."""
        
        print("\n7. Testing MCP Server...")
        
        try:
            # Test MCP server creation
            mcp_server = create_investment_mcp_server()
            assert mcp_server is not None
            
            # Verify components
            assert mcp_server.market_collector is not None
            assert mcp_server.swedish_collector is not None
            assert mcp_server.risk_analyzer is not None
            assert mcp_server.technical_analyzer is not None
            assert mcp_server.portfolio_optimizer is not None
            assert mcp_server.ai_advisor is not None
            
            self.test_results["mcp_server"] = "PASS"
            print("   ✓ MCP server created successfully")
            print("   ✓ All MCP components initialized")
            
        except Exception as e:
            self.test_results["mcp_server"] = f"FAIL: {e}"
            print(f"   ✗ MCP server test failed: {e}")
    
    async def test_data_collection_simulation(self):
        """Test data collection with simulation."""
        
        print("\n8. Testing Data Collection (Simulation)...")
        
        try:
            market_collector = MarketDataCollector()
            
            # Test statistics calculation with mock data
            import pandas as pd
            import numpy as np
            
            np.random.seed(42)
            mock_data = {
                "DNB_GLOBAL_INDEKS_S": pd.DataFrame({
                    "date": pd.date_range("2023-01-01", "2023-12-31", freq="D"),
                    "open": 100 + np.random.randn(365),
                    "high": 105 + np.random.randn(365),
                    "low": 95 + np.random.randn(365),
                    "close": 100 + np.random.randn(365),
                    "volume": np.random.randint(1000, 10000, 365)
                })
            }
            
            statistics = market_collector.get_fund_statistics(mock_data)
            assert "DNB_GLOBAL_INDEKS_S" in statistics
            assert statistics["DNB_GLOBAL_INDEKS_S"]["status"] == "success"
            
            self.test_results["data_collection"] = "PASS"
            print("   ✓ Data collection simulation successful")
            print(f"   ✓ Statistics calculated for {len(statistics)} funds")
            
        except Exception as e:
            self.test_results["data_collection"] = f"FAIL: {e}"
            print(f"   ✗ Data collection test failed: {e}")
    
    def test_analysis_pipeline(self):
        """Test analysis pipeline with mock data."""
        
        print("\n9. Testing Analysis Pipeline...")
        
        try:
            import pandas as pd
            import numpy as np
            
            risk_analyzer = RiskAnalyzer()
            technical_analyzer = TechnicalAnalyzer()
            
            # Create mock data
            np.random.seed(42)
            
            # Mock returns data
            returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns
            risk_metrics = risk_analyzer.calculate_basic_risk_metrics(returns)
            
            # Verify risk metrics
            expected_metrics = [
                "annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"
            ]
            for metric in expected_metrics:
                assert metric in risk_metrics
                assert isinstance(risk_metrics[metric], (int, float))
            
            # Mock market data for technical analysis
            dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
            prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)
            
            market_data = pd.DataFrame({
                "date": dates,
                "close_price": prices,
                "open_price": prices * (1 + np.random.randn(len(dates)) * 0.01),
                "high_price": prices * (1 + np.abs(np.random.randn(len(dates))) * 0.02),
                "low_price": prices * (1 - np.abs(np.random.randn(len(dates))) * 0.02),
                "volume": np.random.randint(1000, 10000, len(dates))
            })
            
            technical_summary = technical_analyzer.generate_technical_summary(market_data)
            assert "current_price" in technical_summary
            
            self.test_results["analysis_pipeline"] = "PASS"
            print("   ✓ Risk analysis pipeline working")
            print("   ✓ Technical analysis pipeline working")
            print(f"   ✓ Calculated {len(risk_metrics)} risk metrics")
            
        except Exception as e:
            self.test_results["analysis_pipeline"] = f"FAIL: {e}"
            print(f"   ✗ Analysis pipeline test failed: {e}")
    
    def test_portfolio_optimization(self):
        """Test portfolio optimization."""
        
        print("\n10. Testing Portfolio Optimization...")
        
        try:
            import pandas as pd
            import numpy as np
            
            portfolio_optimizer = PortfolioOptimizer()
            
            # Create model portfolios
            model_portfolios = portfolio_optimizer.create_model_portfolios()
            
            # Verify model portfolios
            expected_portfolios = ["conservative", "balanced", "growth", "aggressive"]
            for portfolio_name in expected_portfolios:
                assert portfolio_name in model_portfolios
                
                portfolio = model_portfolios[portfolio_name]
                assert "fund_allocation" in portfolio
                
                # Validate allocation
                allocation = portfolio["fund_allocation"]
                validation = validate_fund_allocation(allocation)
                assert validation["valid"], f"Invalid {portfolio_name} portfolio"
            
            # Test optimization with mock data
            np.random.seed(42)
            fund_ids = get_approved_funds()[:5]
            dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
            
            returns_data = {}
            for i, fund_id in enumerate(fund_ids):
                returns_data[fund_id] = np.random.normal(0.0008, 0.02, len(dates))
            
            returns_matrix = pd.DataFrame(returns_data, index=dates)
            
            optimization_result = portfolio_optimizer.mean_variance_optimization(
                returns_matrix, risk_tolerance="medium"
            )
            
            assert optimization_result["success"] is True
            assert "fund_allocation" in optimization_result
            
            # Validate optimized allocation
            optimized_allocation = optimization_result["fund_allocation"]
            validation = validate_fund_allocation(optimized_allocation)
            assert validation["valid"], "Optimized allocation is invalid"
            
            self.test_results["portfolio_optimization"] = "PASS"
            print("   ✓ Model portfolios created successfully")
            print(f"   ✓ {len(model_portfolios)} model portfolios validated")
            print("   ✓ Mean-variance optimization working")
            
        except Exception as e:
            self.test_results["portfolio_optimization"] = f"FAIL: {e}"
            print(f"   ✗ Portfolio optimization test failed: {e}")
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds()
        
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results.values() if r == "PASS"])
        failed_tests = total_tests - passed_tests
        
        print(f"\nExecution Time: {duration:.2f} seconds")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✓" if result == "PASS" else "✗"
            print(f"  {status} {test_name}: {result}")
        
        if failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! The unified Investment MCP system is working correctly.")
        else:
            print(f"\n⚠️  {failed_tests} tests failed. Please review the errors above.")
        
        print("\n" + "=" * 80)
        
        return failed_tests == 0


def main():
    """Main test execution function."""
    
    # Setup logging
    setup_logging(log_level="INFO", enable_file=False)
    
    try:
        # Run system tests
        tester = SystemTester()
        success = tester.run_all_tests()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()