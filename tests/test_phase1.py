"""Phase 1 tests to verify setup."""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.data_collectors.stock_market import StockMarketCollector
from config.settings import settings

def test_environment_setup():
    """Test that environment is properly configured."""
    assert settings is not None
    assert hasattr(settings, 'DEBUG')

def test_stock_market_collector():
    """Test basic functionality of stock market collector."""
    collector = StockMarketCollector()
    assert collector is not None
    
    # Test with a reliable symbol
    data = collector.get_market_data(["AAPL"], period="1d")
    assert isinstance(data, dict)

if __name__ == "__main__":
    # Run basic tests
    test_environment_setup()
    print("✓ Environment setup test passed")
    
    test_stock_market_collector()
    print("✓ Stock market collector test passed")
    
    print("\nPhase 1 setup verification complete!")