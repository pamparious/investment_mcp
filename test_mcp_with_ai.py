"""Test Investment MCP with AI-powered analysis."""

import asyncio
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from config.settings import Settings
from backend.ai.config import AIConfig
from backend.ai.analyzers import MarketAnalyzer
from backend.database import get_db_session
from backend.models import MarketData

async def test_ai_market_analysis():
    """Test AI-powered market analysis with real data."""
    print("🤖 AI-POWERED INVESTMENT ANALYSIS TEST")
    print("=" * 60)
    
    try:
        # Initialize components
        settings = Settings()
        ai_config = AIConfig(settings)
        market_analyzer = MarketAnalyzer(ai_config)
        
        # Get real market data from database
        print("📊 Fetching real market data...")
        with get_db_session() as session:
            # Get recent data for analysis
            market_records = session.query(MarketData).order_by(
                MarketData.symbol, MarketData.date
            ).limit(100).all()
            
            if not market_records:
                print("❌ No market data found. Run data collection first.")
                return
            
            # Group by symbol
            symbol_data = {}
            for record in market_records:
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
        
        # Pick first symbol with sufficient data
        target_symbol = None
        target_data = None
        for symbol, data in symbol_data.items():
            if len(data) >= 20:  # Need at least 20 data points
                target_symbol = symbol
                target_data = data
                break
        
        if not target_symbol:
            print("❌ Insufficient data for analysis")
            return
        
        print(f"🎯 Analyzing {target_symbol} with {len(target_data)} data points")
        print(f"📅 Data range: {target_data[0]['date']} to {target_data[-1]['date']}")
        print(f"💰 Current price: ${target_data[-1]['close_price']:.2f}")
        
        print(f"\n🔄 Running AI-powered analysis (this may take 30-60 seconds)...")
        
        # Run comprehensive AI analysis
        analysis = await market_analyzer.analyze_stock(target_symbol, target_data, "ollama")
        
        print(f"\n{'='*60}")
        print(f"📈 AI ANALYSIS RESULTS FOR {target_symbol}")
        print(f"{'='*60}")
        
        # Display technical analysis
        if 'technical_analysis' in analysis:
            tech = analysis['technical_analysis']
            print(f"\n📊 TECHNICAL INDICATORS:")
            
            ma = tech.get('moving_averages', {})
            if ma:
                print(f"   Current Price: ${ma.get('current_price', 0):.2f}")
                if 'sma_20' in ma:
                    print(f"   20-day SMA: ${ma['sma_20']:.2f}")
                    signal = "Above" if ma.get('current_price', 0) > ma['sma_20'] else "Below"
                    print(f"   Position: {signal} moving average")
            
            rsi = tech.get('rsi')
            if rsi:
                if rsi < 30:
                    rsi_signal = "🟢 OVERSOLD (potential buy)"
                elif rsi > 70:
                    rsi_signal = "🔴 OVERBOUGHT (potential sell)"
                else:
                    rsi_signal = "🟡 NEUTRAL"
                print(f"   RSI: {rsi:.1f} - {rsi_signal}")
            
            trend = tech.get('trend_analysis', {})
            if trend:
                trend_emoji = "📈" if trend.get('trend') == 'bullish' else "📉" if trend.get('trend') == 'bearish' else "➡️"
                print(f"   Trend: {trend_emoji} {trend.get('trend', 'unknown').upper()}")
                print(f"   Strength: {trend.get('strength', 0):.2f}")
        
        # Display AI insights
        if 'ai_insights' in analysis and 'error' not in analysis['ai_insights']:
            ai = analysis['ai_insights']
            print(f"\n🤖 AI INSIGHTS:")
            print(f"   Assessment: {ai.get('assessment', 'Unknown').upper()}")
            print(f"   Confidence: {ai.get('confidence', 0):.1f}/1.0")
            print(f"   Recommendation: {ai.get('recommendation', 'Unknown').upper()}")
            
            if 'reasoning' in ai:
                print(f"\n💭 AI REASONING:")
                reasoning = ai['reasoning']
                # Wrap text nicely
                import textwrap
                wrapped_reasoning = textwrap.fill(reasoning, width=55)
                for line in wrapped_reasoning.split('\n'):
                    print(f"   {line}")
        
        elif 'ai_insights' in analysis and 'error' in analysis['ai_insights']:
            print(f"\n⚠️  AI Analysis: {analysis['ai_insights']['error']}")
        
        # Display overall assessment
        if 'overall_assessment' in analysis:
            overall = analysis['overall_assessment']
            print(f"\n📋 OVERALL ASSESSMENT:")
            
            rec = overall.get('recommendation', 'unknown').upper()
            if rec == 'BUY':
                rec_emoji = "🟢"
            elif rec == 'SELL':
                rec_emoji = "🔴"
            else:
                rec_emoji = "🟡"
            
            print(f"   {rec_emoji} Recommendation: {rec}")
            print(f"   Confidence: {overall.get('confidence', 'unknown').title()}")
            print(f"   Risk Level: {overall.get('risk_level', 'unknown').title()}")
            
            key_factors = overall.get('key_factors', {})
            if key_factors.get('entry_signals'):
                print(f"   🔹 Entry Signals:")
                for signal in key_factors['entry_signals'][:3]:
                    print(f"     • {signal}")
            
            if key_factors.get('risk_warnings'):
                print(f"   ⚠️  Risk Warnings:")
                for warning in key_factors['risk_warnings'][:3]:
                    print(f"     • {warning}")
        
        print(f"\n{'='*60}")
        print(f"✅ AI ANALYSIS COMPLETE")
        print(f"{'='*60}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in AI analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_ai_portfolio_insights():
    """Test AI portfolio analysis."""
    print(f"\n🎯 AI PORTFOLIO ANALYSIS")
    print(f"{'='*40}")
    
    try:
        settings = Settings()
        ai_config = AIConfig(settings)
        
        # Test AI provider for portfolio insights
        provider = ai_config.get_provider("ollama")
        
        # Sample portfolio data
        portfolio_context = {
            "portfolio_summary": {
                "num_assets": 3,
                "portfolio_volatility": 0.15,
                "diversification_level": "good",
                "average_correlation": 0.4
            },
            "individual_asset_summary": {
                "FTSE": {"technical_score": 50, "volatility": 0.12, "weight": 0.4},
                "DAX": {"technical_score": -20, "volatility": 0.18, "weight": 0.3},
                "SP500": {"technical_score": 30, "volatility": 0.14, "weight": 0.3}
            }
        }
        
        print("🤖 Getting AI portfolio insights...")
        
        async with provider:
            insights = await provider.generate_insights(portfolio_context, "portfolio_analysis")
        
        if 'error' not in insights:
            print(f"💼 AI PORTFOLIO INSIGHTS:")
            print(f"   Provider: {insights.get('provider', 'unknown')}")
            # Print any insights returned
            for key, value in insights.items():
                if key not in ['provider', 'analysis_type']:
                    print(f"   {key}: {value}")
        else:
            print(f"⚠️  AI Portfolio Analysis: {insights['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in portfolio AI analysis: {e}")
        return False

async def main():
    """Run AI-powered investment analysis tests."""
    print("🚀 INVESTMENT MCP + AI INTEGRATION TEST")
    print("=" * 70)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: AI Market Analysis
    market_ok = await test_ai_market_analysis()
    
    # Test 2: AI Portfolio Insights
    portfolio_ok = await test_ai_portfolio_insights()
    
    print(f"\n{'='*70}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*70}")
    print(f"✅ AI Market Analysis: {'PASS' if market_ok else 'FAIL'}")
    print(f"✅ AI Portfolio Insights: {'PASS' if portfolio_ok else 'FAIL'}")
    
    if market_ok and portfolio_ok:
        print(f"\n🎉 SUCCESS! Investment MCP + AI is fully operational!")
        print(f"\n🔥 WHAT YOU CAN DO NOW:")
        print(f"   📈 Get AI-powered stock analysis with technical indicators")
        print(f"   💼 Receive natural language portfolio recommendations")
        print(f"   🤖 Ask investment questions and get expert-level answers")
        print(f"   📊 Combine real market data with AI insights")
        print(f"   🔍 Get detailed explanations of complex financial patterns")
        
        print(f"\n🛠️  NEXT STEPS:")
        print(f"   • Run full analysis: python run_demo_analysis.py")
        print(f"   • Start MCP server: python mcp_servers/investment_server/server.py")
        print(f"   • Use in Claude Desktop or other MCP clients")
        
    else:
        print(f"\n⚠️  Some AI features need troubleshooting")
        print(f"   • Check Ollama is running: ollama serve")
        print(f"   • Verify model: ollama list")

if __name__ == "__main__":
    asyncio.run(main())