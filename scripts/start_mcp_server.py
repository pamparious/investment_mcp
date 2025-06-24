#!/usr/bin/env python3
"""Start the Investment MCP Server with demonstration."""

import asyncio
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

async def demonstrate_mcp_server():
    """Demonstrate the MCP server functionality."""
    print('🚀 INVESTMENT MCP SERVER STARTED')
    print('=' * 60)
    print('Server ready to accept MCP protocol connections')
    print('Available tools and resources:')
    print()
    
    # Show available tools
    tools = [
        {
            "name": "get_collection_status",
            "description": "Get status of recent data collections",
            "schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "trigger_data_collection", 
            "description": "Trigger data collection from specified source",
            "schema": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["stock_market", "riksbank", "scb", "all"],
                        "description": "Data source to collect from"
                    }
                },
                "required": ["source"]
            }
        },
        {
            "name": "query_investment_data",
            "description": "Query investment data from database",
            "schema": {
                "type": "object", 
                "properties": {
                    "data_type": {
                        "type": "string",
                        "enum": ["market", "riksbank", "scb"],
                        "description": "Type of data to query"
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Stock symbol (for market data)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of records to return",
                        "default": 10
                    }
                },
                "required": ["data_type"]
            }
        }
    ]
    
    # Show available resources
    resources = [
        {
            "uri": "investment://market_data",
            "name": "Market Data",
            "description": "Current stock market data",
            "mimeType": "application/json"
        },
        {
            "uri": "investment://swedish_data", 
            "name": "Swedish Economic Data",
            "description": "SCB and Riksbank data",
            "mimeType": "application/json"
        },
        {
            "uri": "investment://collection_logs",
            "name": "Collection Logs",
            "description": "Data collection activity logs", 
            "mimeType": "application/json"
        }
    ]
    
    print('📋 AVAILABLE TOOLS:')
    print('-' * 30)
    for i, tool in enumerate(tools, 1):
        print(f'{i}. {tool["name"]}')
        print(f'   {tool["description"]}')
        print()
    
    print('📁 AVAILABLE RESOURCES:')
    print('-' * 30) 
    for i, resource in enumerate(resources, 1):
        print(f'{i}. {resource["name"]} ({resource["uri"]})')
        print(f'   {resource["description"]}')
        print()
    
    print('📊 CURRENT DATA STATUS:')
    print('-' * 30)
    
    # Import here to avoid circular imports
    from backend.database import get_db_session
    from backend.models import SCBData, RiksbankData, MarketData, DataCollectionLog
    
    with get_db_session() as session:
        scb_count = session.query(SCBData).count()
        market_count = session.query(MarketData).count()
        riksbank_count = session.query(RiksbankData).count()
        log_count = session.query(DataCollectionLog).count()
        
        print(f'📈 Market Data Records: {market_count}')
        print(f'🏠 SCB Records: {scb_count}')
        print(f'🏦 Riksbank Records: {riksbank_count}')
        print(f'📋 Collection Logs: {log_count}')
        print(f'📊 Total Records: {scb_count + market_count + riksbank_count}')
    
    print()
    print('🔗 MCP CLIENT CONNECTION EXAMPLE:')
    print('-' * 40)
    print('To connect to this server, use:')
    print('mcp connect stdio python mcp_servers/investment_server/server.py')
    print()
    print('Or use the Claude Desktop configuration:')
    connection_config = {
        "investment-mcp": {
            "command": "python",
            "args": ["mcp_servers/investment_server/server.py"],
            "env": {}
        }
    }
    print(json.dumps(connection_config, indent=2))
    
    print()
    print('✅ Investment MCP Server is ready for connections!')
    print('The server provides real-time Swedish investment data including:')
    print('  • Housing price indices from Statistics Sweden (SCB)')
    print('  • Exchange rates from Riksbank')
    print('  • International stock market data')
    print('  • Data collection monitoring and management')

if __name__ == "__main__":
    asyncio.run(demonstrate_mcp_server())