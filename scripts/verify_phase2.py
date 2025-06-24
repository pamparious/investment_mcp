#!/usr/bin/env python3
"""Comprehensive Phase 2 verification script."""

import asyncio
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.database import init_database, get_db_session
from backend.models import Base, MarketData, RiksbankData, SCBData, DataCollectionLog
from backend.mcp_agents.data_agent import DataCollectionAgent
from backend.data_collectors.stock_market import StockMarketCollector
from backend.data_collectors.riksbanken import RiksbankCollector
from backend.data_collectors.scb import SCBCollector
from config.settings import settings

async def verify_database():
    """Verify database setup."""
    print("🗄️  Verifying database setup...")
    
    try:
        init_database()
        print("✅ Database initialized successfully")
        
        # Test database connection
        with get_db_session() as session:
            # Test each table
            tables = [MarketData, RiksbankData, SCBData, DataCollectionLog]
            for table in tables:
                count = session.query(table).count()
                print(f"   📊 {table.__tablename__}: {count} records")
        
        return True
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

async def verify_data_collectors():
    """Verify data collector functionality."""
    print("\n🔄 Verifying data collectors...")
    
    results = {}
    
    # Test Stock Market Collector
    try:
        collector = StockMarketCollector()
        data = collector.get_market_data(["^OMX"], period="1d")
        results['stock_market'] = len(data) > 0
        print(f"   📈 Stock Market: {'✅ Working' if results['stock_market'] else '⚠️  No data'}")
    except Exception as e:
        results['stock_market'] = False
        print(f"   📈 Stock Market: ❌ Error - {e}")
    
    # Test Riksbank Collector
    try:
        async with RiksbankCollector() as collector:
            data = await collector.fetch_series_data("SEKREPORATE")
            results['riksbank'] = True  # API connection successful
            print(f"   🏦 Riksbank: ✅ API accessible")
    except Exception as e:
        results['riksbank'] = False
        print(f"   🏦 Riksbank: ❌ Error - {e}")
    
    # Test SCB Collector
    try:
        async with SCBCollector() as collector:
            metadata = await collector.get_table_metadata("HE0103T04")
            results['scb'] = isinstance(metadata, dict)
            print(f"   📋 SCB: {'✅ API accessible' if results['scb'] else '⚠️  Limited access'}")
    except Exception as e:
        results['scb'] = False
        print(f"   📋 SCB: ❌ Error - {e}")
    
    return results

async def verify_mcp_agent():
    """Verify MCP agent functionality."""
    print("\n🤖 Verifying MCP agent...")
    
    try:
        agent = DataCollectionAgent()
        
        # Test status retrieval
        status = await agent.get_collection_status()
        print("   ✅ Agent initialized successfully")
        print(f"   📊 Collections in last 24h: {status['total_collections_24h']}")
        
        return True
    except Exception as e:
        print(f"   ❌ Agent error: {e}")
        return False

def verify_configuration():
    """Verify configuration setup."""
    print("\n⚙️  Verifying configuration...")
    
    config_items = [
        ("Database URL", settings.DATABASE_URL),
        ("Riksbank API Base", settings.RIKSBANKEN_API_BASE),
        ("SCB API Base", settings.SCB_API_BASE),
        ("Data Collection Interval", settings.DATA_COLLECTION_INTERVAL),
        ("Max API Calls/Min", settings.MAX_API_CALLS_PER_MINUTE),
    ]
    
    for name, value in config_items:
        status = "✅" if value else "⚠️"
        print(f"   {status} {name}: {value}")
    
    # Check optional API keys
    optional_keys = [
        ("Alpha Vantage API Key", settings.ALPHA_VANTAGE_API_KEY),
        ("Riksbank API Key", settings.RIKSBANKEN_API_KEY),
        ("SCB API Key", settings.SCB_API_KEY),
    ]
    
    for name, value in optional_keys:
        status = "✅" if value else "💡"
        display_value = "***configured***" if value else "not set (optional)"
        print(f"   {status} {name}: {display_value}")

def verify_file_structure():
    """Verify file structure is complete."""
    print("\n📁 Verifying file structure...")
    
    required_files = [
        "backend/data_collectors/riksbanken.py",
        "backend/data_collectors/scb.py",
        "backend/mcp_agents/data_agent.py",
        "backend/models/__init__.py",
        "backend/database.py",
        "mcp_servers/investment_server/server.py",
        "config/settings.py",
        "tests/test_phase2.py",
    ]
    
    all_present = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
            all_present = False
    
    return all_present

async def main():
    """Run complete Phase 2 verification."""
    print("🚀 Investment MCP Phase 2 Verification")
    print("=" * 50)
    
    # Verify file structure
    files_ok = verify_file_structure()
    
    # Verify configuration
    verify_configuration()
    
    # Verify database
    db_ok = await verify_database()
    
    # Verify data collectors
    collectors_results = await verify_data_collectors()
    
    # Verify MCP agent
    agent_ok = await verify_mcp_agent()
    
    # Summary
    print("\n📋 Verification Summary")
    print("=" * 30)
    
    checks = [
        ("File Structure", files_ok),
        ("Database Setup", db_ok),
        ("Stock Market Collector", collectors_results.get('stock_market', False)),
        ("Riksbank Collector", collectors_results.get('riksbank', False)),
        ("SCB Collector", collectors_results.get('scb', False)),
        ("MCP Agent", agent_ok),
    ]
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    for name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 Phase 2 setup is complete and working!")
        print("\nNext steps:")
        print("1. Add API keys to .env file if needed")
        print("2. Test full data collection: python backend/mcp_agents/data_agent.py --source all")
        print("3. Start MCP server: python mcp_servers/investment_server/server.py")
        print("4. Proceed to Phase 3: Analysis & AI Integration")
    else:
        print(f"\n⚠️  {total - passed} issues found. Please address them before proceeding.")

if __name__ == "__main__":
    asyncio.run(main())