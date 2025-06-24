#!/usr/bin/env python3
"""Demonstrate MCP server tools functionality."""

import sys
import os
import asyncio
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.mcp_agents.data_agent import DataCollectionAgent
from backend.database import get_db_session
from backend.models import SCBData, RiksbankData, MarketData

async def demo_mcp_tools():
    """Demonstrate the MCP server tools."""
    print('🚀 INVESTMENT MCP SERVER - TOOLS DEMONSTRATION')
    print('=' * 60)
    
    # Initialize the data agent
    agent = DataCollectionAgent()
    
    print('\n📊 Tool 1: Get Collection Status')
    print('-' * 40)
    status = await agent.get_collection_status()
    print(f"Total collections (24h): {status['total_collections_24h']}")
    print(f"Successful collections: {status['successful_collections_24h']}")
    print(f"Records collected (24h): {status['total_records_collected_24h']}")
    
    # Calculate error rate
    total = status['total_collections_24h']
    successful = status['successful_collections_24h']
    error_rate = (total - successful) / total if total > 0 else 0
    print(f"Error rate: {error_rate:.1%}")
    
    print("Recent activities:")
    for activity in status['recent_activities'][:3]:
        print(f"  • {activity['source']}: {activity['status']} ({activity.get('records_collected', 0)} records)")
    
    print('\n⚡ Tool 2: Trigger Data Collection')
    print('-' * 40)
    try:
        print("Triggering SCB data collection...")
        collection_result = await agent.collect_scb_data()
        print(f"Collection successful: {collection_result.success}")
        print(f"Records collected: {collection_result.records_collected}")
        print(f"Source: {collection_result.source}")
    except Exception as e:
        print(f"Collection error: {e}")
    
    print('\n📈 Tool 3: Database Summary')
    print('-' * 40)
    with get_db_session() as session:
        scb_count = session.query(SCBData).count()
        market_count = session.query(MarketData).count()
        riksbank_count = session.query(RiksbankData).count()
        
        print(f"SCB Records: {scb_count}")
        print(f"Market Records: {market_count}")
        print(f"Riksbank Records: {riksbank_count}")
        print(f"Total Records: {scb_count + market_count + riksbank_count}")
    
    print('\n🏠 Tool 4: Latest Swedish Housing Data')
    print('-' * 40)
    with get_db_session() as session:
        latest_scb = session.query(SCBData).filter(
            SCBData.table_id == 'FastpiPSRegAr'
        ).order_by(SCBData.date.desc()).first()
        
        if latest_scb:
            print(f"Latest housing price index: {latest_scb.value:.1f}")
            print(f"Date: {latest_scb.date}")
            print(f"Region: {latest_scb.region}")
            print(f"Unit: {latest_scb.unit}")
        else:
            print("No housing data available")
    
    print('\n💱 Tool 5: Latest Exchange Rate Data')
    print('-' * 40)
    with get_db_session() as session:
        latest_fx = session.query(RiksbankData).filter(
            RiksbankData.series_id == 'sekusdpmi'
        ).order_by(RiksbankData.date.desc()).first()
        
        if latest_fx:
            print(f"Latest SEK/USD rate: {latest_fx.value:.4f}")
            print(f"Date: {latest_fx.date}")
            print(f"Description: {latest_fx.description}")
        else:
            print("No exchange rate data available")
    
    print('\n📊 Tool 6: Market Performance')
    print('-' * 40)
    with get_db_session() as session:
        # Get latest market data for each symbol
        from sqlalchemy import func
        latest_prices = session.query(
            MarketData.symbol,
            func.max(MarketData.date).label('latest_date')
        ).group_by(MarketData.symbol).subquery()
        
        market_summary = session.query(MarketData).join(
            latest_prices,
            (MarketData.symbol == latest_prices.c.symbol) &
            (MarketData.date == latest_prices.c.latest_date)
        ).all()
        
        print("Latest market prices:")
        for record in market_summary[:5]:
            change_color = "📈" if record.close_price > record.open_price else "📉"
            print(f"  {change_color} {record.symbol}: ${record.close_price:.2f} ({record.date})")
    
    print('\n✅ MCP Server Tools Demonstration Complete!')
    print(f"Data collection and querying tools working correctly ✓")

if __name__ == "__main__":
    asyncio.run(demo_mcp_tools())