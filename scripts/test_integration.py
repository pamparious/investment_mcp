#!/usr/bin/env python3
"""Integration test script for Phase 2."""

import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.utils.logging_config import setup_logging
from backend.database import init_database
from backend.mcp_agents.data_agent import DataCollectionAgent

async def main():
    """Run integration tests."""
    print("🚀 Starting Investment MCP Phase 2 Integration Test\n")
    
    # Setup logging
    setup_logging()
    logging.info("Integration test started")
    
    # Initialize database
    print("📊 Initializing database...")
    init_database()
    print("✅ Database initialized\n")
    
    # Initialize data agent
    print("🤖 Creating data collection agent...")
    agent = DataCollectionAgent()
    print("✅ Data agent created\n")
    
    # Test 1: Collection status
    print("📈 Testing collection status...")
    status = await agent.get_collection_status()
    print(f"✅ Status retrieved: {status['total_collections_24h']} collections in last 24h\n")
    
    # Test 2: Stock data collection
    print("📊 Testing stock data collection...")
    try:
        result = await agent.collect_stock_data(["^OMX"])
        print(f"✅ Stock data: {result.records_collected} records, Success: {result.success}\n")
    except Exception as e:
        print(f"⚠️  Stock data collection failed: {e}\n")
    
    # Test 3: Riksbank data collection  
    print("🏦 Testing Riksbank data collection...")
    try:
        result = await agent.collect_riksbank_data(days_back=7)
        print(f"✅ Riksbank data: {result.records_collected} records, Success: {result.success}\n")
    except Exception as e:
        print(f"⚠️  Riksbank data collection failed: {e}\n")
    
    # Test 4: SCB data collection
    print("📋 Testing SCB data collection...")
    try:
        result = await agent.collect_scb_data(years_back=1)
        print(f"✅ SCB data: {result.records_collected} records, Success: {result.success}\n")
    except Exception as e:
        print(f"⚠️  SCB data collection failed: {e}\n")
    
    # Final status check
    print("📊 Final status check...")
    final_status = await agent.get_collection_status()
    print(f"✅ Final collections: {final_status['total_collections_24h']}")
    print(f"✅ Total records: {final_status['total_records_collected_24h']}")
    
    print("\n🎉 Integration test completed!")

if __name__ == "__main__":
    asyncio.run(main())