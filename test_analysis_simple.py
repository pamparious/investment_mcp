"""Simple test for investment analysis functionality."""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from config.settings import Settings
from backend.ai.config import AIConfig
from backend.database import get_db_session
from backend.models import MarketData, RiksbankData, SCBData

async def test_ai_providers():
    """Test AI provider availability."""
    print("=" * 50)
    print("TESTING AI PROVIDERS")
    print("=" * 50)
    
    try:
        settings = Settings()
        ai_config = AIConfig(settings)
        
        # Get list of available providers
        available_providers = ai_config.list_available_providers()
        print(f"Available providers: {available_providers}")
        
        # Test each provider
        for provider_name, is_installed in available_providers.items():
            print(f"\nTesting {provider_name}:")
            if is_installed:
                try:
                    test_result = await ai_config.test_provider(provider_name)
                    print(f"  Status: {'Available' if test_result.get('available', False) else 'Not Available'}")
                    if test_result.get('error'):
                        print(f"  Error: {test_result['error']}")
                    if test_result.get('model'):
                        print(f"  Model: {test_result['model']}")
                except Exception as e:
                    print(f"  Error testing: {e}")
            else:
                print(f"  Status: Library not installed")
        
        return True
        
    except Exception as e:
        print(f"Error testing providers: {e}")
        return False

def test_data_availability():
    """Test what data is available in the database."""
    print("\n" + "=" * 50)
    print("CHECKING DATA AVAILABILITY")
    print("=" * 50)
    
    try:
        with get_db_session() as session:
            # Count market data
            market_count = session.query(MarketData).count()
            print(f"Market Data Records: {market_count}")
            
            if market_count > 0:
                # Get sample of market data
                sample_market = session.query(MarketData).limit(3).all()
                print("Sample Market Data:")
                for record in sample_market:
                    print(f"  {record.symbol} - {record.date} - Close: {record.close_price}")
            
            # Count Riksbank data
            riksbank_count = session.query(RiksbankData).count()
            print(f"\nRiksbank Data Records: {riksbank_count}")
            
            if riksbank_count > 0:
                # Get sample of Riksbank data
                sample_riksbank = session.query(RiksbankData).limit(3).all()
                print("Sample Riksbank Data:")
                for record in sample_riksbank:
                    print(f"  {record.series_id} - {record.date} - Value: {record.value}")
            
            # Count SCB data
            scb_count = session.query(SCBData).count()
            print(f"\nSCB Data Records: {scb_count}")
            
            if scb_count > 0:
                # Get sample of SCB data
                sample_scb = session.query(SCBData).limit(3).all()
                print("Sample SCB Data:")
                for record in sample_scb:
                    print(f"  {record.table_id} - {record.date} - Value: {record.value}")
            
            return market_count > 0 or riksbank_count > 0 or scb_count > 0
            
    except Exception as e:
        print(f"Error checking data: {e}")
        return False

async def test_technical_analysis():
    """Test technical analysis functionality."""
    print("\n" + "=" * 50)
    print("TESTING TECHNICAL ANALYSIS")
    print("=" * 50)
    
    try:
        from backend.analysis.patterns import TechnicalAnalyzer
        
        # Get some market data for analysis
        with get_db_session() as session:
            # Get recent market data for a symbol
            cutoff_date = datetime.now() - timedelta(days=100)
            market_data = session.query(MarketData).filter(
                MarketData.date >= cutoff_date
            ).order_by(MarketData.symbol, MarketData.date).limit(50).all()
            
            if not market_data:
                print("No market data available for analysis")
                return False
            
            # Group by symbol
            symbol_data = {}
            for record in market_data:
                if record.symbol not in symbol_data:
                    symbol_data[record.symbol] = []
                
                symbol_data[record.symbol].append({
                    "date": record.date.isoformat(),
                    "open_price": float(record.open_price),
                    "high_price": float(record.high_price),
                    "low_price": float(record.low_price),
                    "close_price": float(record.close_price),
                    "volume": int(record.volume)
                })
            
            if not symbol_data:
                print("No symbol data available")
                return False
            
            # Test technical analysis on first symbol
            first_symbol = list(symbol_data.keys())[0]
            data = symbol_data[first_symbol]
            
            print(f"Analyzing {first_symbol} with {len(data)} data points...")
            
            analyzer = TechnicalAnalyzer()
            analysis = analyzer.analyze_symbol(first_symbol, data)
            
            print("Technical Analysis Results:")
            print(f"  Symbol: {analysis.get('symbol', 'Unknown')}")
            print(f"  Data Points: {analysis.get('data_points', 0)}")
            
            # Moving averages
            ma = analysis.get('moving_averages', {})
            if ma:
                print("  Moving Averages:")
                if 'current_price' in ma:
                    print(f"    Current Price: {ma['current_price']:.2f}")
                if 'sma_20' in ma:
                    print(f"    SMA 20: {ma['sma_20']:.2f}")
                if 'sma_50' in ma:
                    print(f"    SMA 50: {ma['sma_50']:.2f}")
            
            # Trend analysis
            trend = analysis.get('trend_analysis', {})
            if trend:
                print("  Trend Analysis:")
                print(f"    Trend: {trend.get('trend', 'Unknown')}")
                print(f"    Strength: {trend.get('strength', 0):.2f}")
                print(f"    Confidence: {trend.get('confidence', 'Unknown')}")
            
            # RSI
            rsi = analysis.get('rsi')
            if rsi:
                print(f"  RSI: {rsi:.2f}")
            
            # Technical score
            tech_score = analysis.get('technical_score', {})
            if tech_score:
                print("  Technical Score:")
                print(f"    Score: {tech_score.get('score', 0)}")
                print(f"    Signal: {tech_score.get('signal', 'neutral')}")
                print(f"    Signals: {tech_score.get('signals', [])}")
            
            return True
            
    except Exception as e:
        print(f"Error in technical analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_risk_analysis():
    """Test risk analysis functionality."""
    print("\n" + "=" * 50)
    print("TESTING RISK ANALYSIS")
    print("=" * 50)
    
    try:
        from backend.analysis.risk_metrics import RiskAnalyzer
        
        # Get market data
        with get_db_session() as session:
            cutoff_date = datetime.now() - timedelta(days=100)
            market_data = session.query(MarketData).filter(
                MarketData.date >= cutoff_date
            ).order_by(MarketData.symbol, MarketData.date).limit(50).all()
            
            if not market_data:
                print("No market data available for risk analysis")
                return False
            
            # Convert to list format
            data_list = []
            for record in market_data:
                data_list.append({
                    "date": record.date.isoformat(),
                    "close_price": float(record.close_price)
                })
            
            print(f"Analyzing risk with {len(data_list)} data points...")
            
            analyzer = RiskAnalyzer()
            
            # Test volatility calculation
            vol_results = analyzer.calculate_volatility(data_list)
            if 'error' not in vol_results:
                print("Volatility Analysis:")
                print(f"  Historical Volatility: {vol_results.get('historical_volatility', 0):.4f}")
                print(f"  Volatility Class: {vol_results.get('volatility_class', 'unknown')}")
                print(f"  Risk Level: {vol_results.get('risk_level', 'unknown')}")
            
            # Test VaR calculation
            var_results = analyzer.value_at_risk(data_list)
            if 'error' not in var_results:
                print("\nValue at Risk (VaR) Analysis:")
                print(f"  5% VaR: {var_results.get('historical_var', 0):.4f}")
                print(f"  Risk Assessment: {var_results.get('risk_assessment', 'unknown')}")
            
            # Test maximum drawdown
            dd_results = analyzer.maximum_drawdown(data_list)
            if 'error' not in dd_results:
                print("\nMaximum Drawdown Analysis:")
                print(f"  Max Drawdown: {dd_results.get('max_dd_percentage', 0):.2f}%")
                print(f"  Risk Level: {dd_results.get('risk_level', 'unknown')}")
                print(f"  Current Drawdown: {dd_results.get('current_dd_percentage', 0):.2f}%")
            
            return True
            
    except Exception as e:
        print(f"Error in risk analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("INVESTMENT MCP ANALYSIS TESTING")
    print("=" * 70)
    
    # Test 1: AI Providers
    providers_ok = await test_ai_providers()
    
    # Test 2: Data Availability
    data_ok = test_data_availability()
    
    # Test 3: Technical Analysis (only if data available)
    technical_ok = False
    if data_ok:
        technical_ok = await test_technical_analysis()
    
    # Test 4: Risk Analysis (only if data available)
    risk_ok = False
    if data_ok:
        risk_ok = await test_risk_analysis()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"AI Providers Test: {'PASS' if providers_ok else 'FAIL'}")
    print(f"Data Availability Test: {'PASS' if data_ok else 'FAIL'}")
    print(f"Technical Analysis Test: {'PASS' if technical_ok else 'FAIL' if data_ok else 'SKIP (no data)'}")
    print(f"Risk Analysis Test: {'PASS' if risk_ok else 'FAIL' if data_ok else 'SKIP (no data)'}")
    
    overall_success = providers_ok and data_ok and (technical_ok or not data_ok) and (risk_ok or not data_ok)
    print(f"\nOVERALL: {'SUCCESS' if overall_success else 'SOME ISSUES FOUND'}")
    
    if not overall_success:
        print("\nNext steps:")
        if not providers_ok:
            print("- Set up at least one AI provider (Ollama recommended for local testing)")
        if not data_ok:
            print("- Run data collection: python backend/mcp_agents/data_agent.py --source all")
        if data_ok and not technical_ok:
            print("- Check technical analysis implementation")
        if data_ok and not risk_ok:
            print("- Check risk analysis implementation")

if __name__ == "__main__":
    asyncio.run(main())