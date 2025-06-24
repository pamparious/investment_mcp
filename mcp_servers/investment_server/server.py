import asyncio
import logging
import sys
import os
from mcp.server.stdio import stdio_server
from mcp.server import NotificationOptions, Server
from mcp.types import Resource, Tool, TextContent
from typing import Any
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.mcp_agents.data_agent import DataCollectionAgent
from backend.database import init_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("investment-mcp-server")

# Create server instance
server = Server("investment-mcp-server")

# Initialize data collection agent
data_agent = DataCollectionAgent()

@server.list_resources()
async def handle_list_resources():
    """List available resources."""
    return [
        Resource(
            uri="investment://market-data",
            name="Market Data",
            description="Current stock market data and indices",
            mimeType="application/json",
        ),
        Resource(
            uri="investment://riksbank-data",
            name="Riksbank Data",
            description="Swedish central bank data including interest rates",
            mimeType="application/json",
        ),
        Resource(
            uri="investment://scb-data",
            name="SCB Data",
            description="Statistics Sweden data including housing prices",
            mimeType="application/json",
        ),
        Resource(
            uri="investment://collection-status",
            name="Collection Status",
            description="Status of recent data collection activities",
            mimeType="application/json",
        ),
    ]

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read a specific resource."""
    if uri == "investment://collection-status":
        status = await data_agent.get_collection_status()
        return json.dumps(status, indent=2)
    elif uri == "investment://market-data":
        return json.dumps({"message": "Use collect_market_data tool to fetch current data"})
    elif uri == "investment://riksbank-data":
        return json.dumps({"message": "Use collect_riksbank_data tool to fetch current data"})
    elif uri == "investment://scb-data":
        return json.dumps({"message": "Use collect_scb_data tool to fetch current data"})
    else:
        raise ValueError(f"Unknown resource: {uri}")

@server.list_tools()
async def handle_list_tools():
    """List available tools."""
    return [
        Tool(
            name="collect_market_data",
            description="Collect current stock market data for specified symbols",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock symbols to collect (optional)",
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="collect_riksbank_data",
            description="Collect data from Riksbank (Swedish central bank)",
            inputSchema={
                "type": "object",
                "properties": {
                    "days_back": {
                        "type": "integer",
                        "description": "Number of days back to collect data",
                        "default": 30
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="collect_scb_data",
            description="Collect data from SCB (Statistics Sweden)",
            inputSchema={
                "type": "object",
                "properties": {
                    "years_back": {
                        "type": "integer",
                        "description": "Number of years back to collect data",
                        "default": 2
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="collect_all_data",
            description="Collect data from all sources (market, Riksbank, SCB)",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="get_collection_status",
            description="Get status of recent data collection activities",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]):
    """Handle tool calls."""
    try:
        if name == "collect_market_data":
            symbols = arguments.get("symbols")
            result = await data_agent.collect_stock_data(symbols)
            
            return [TextContent(
                type="text",
                text=f"Market data collection completed:\n"
                     f"Status: {'Success' if result.success else 'Failed'}\n"
                     f"Records collected: {result.records_collected}\n"
                     f"Source: {result.source}\n"
                     f"Error: {result.error_message or 'None'}"
            )]
            
        elif name == "collect_riksbank_data":
            days_back = arguments.get("days_back", 30)
            result = await data_agent.collect_riksbank_data(days_back)
            
            return [TextContent(
                type="text",
                text=f"Riksbank data collection completed:\n"
                     f"Status: {'Success' if result.success else 'Failed'}\n"
                     f"Records collected: {result.records_collected}\n"
                     f"Days back: {days_back}\n"
                     f"Error: {result.error_message or 'None'}"
            )]
            
        elif name == "collect_scb_data":
            years_back = arguments.get("years_back", 2)
            result = await data_agent.collect_scb_data(years_back)
            
            return [TextContent(
                type="text",
                text=f"SCB data collection completed:\n"
                     f"Status: {'Success' if result.success else 'Failed'}\n"
                     f"Records collected: {result.records_collected}\n"
                     f"Years back: {years_back}\n"
                     f"Error: {result.error_message or 'None'}"
            )]
            
        elif name == "collect_all_data":
            results = await data_agent.collect_all_data()
            
            summary = "Comprehensive data collection completed:\n\n"
            for result in results:
                summary += f"• {result.source.upper()}:\n"
                summary += f"  Status: {'Success' if result.success else 'Failed'}\n"
                summary += f"  Records: {result.records_collected}\n"
                if result.error_message:
                    summary += f"  Error: {result.error_message}\n"
                summary += "\n"
            
            total_records = sum(r.records_collected for r in results)
            successful = sum(1 for r in results if r.success)
            summary += f"Total: {successful}/{len(results)} successful, {total_records} records collected"
            
            return [TextContent(
                type="text",
                text=summary
            )]
            
        elif name == "get_collection_status":
            status = await data_agent.get_collection_status()
            
            status_text = f"Data Collection Status (Last 24 hours):\n\n"
            status_text += f"Total collections: {status['total_collections_24h']}\n"
            status_text += f"Successful: {status['successful_collections_24h']}\n"
            status_text += f"Total records: {status['total_records_collected_24h']}\n\n"
            
            status_text += "Recent Activities:\n"
            for activity in status['recent_activities'][:5]:
                status_text += f"• {activity['source']} ({activity['type']}): "
                status_text += f"{activity['status']} - {activity['records']} records\n"
                status_text += f"  Time: {activity['timestamp']}\n"
                if activity.get('error'):
                    status_text += f"  Error: {activity['error']}\n"
            
            return [TextContent(
                type="text",
                text=status_text
            )]
            
        else:
            raise ValueError(f"Unknown tool: {name}")
            
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return [TextContent(
            type="text",
            text=f"Error executing {name}: {str(e)}"
        )]

async def main():
    """Run the MCP server."""
    # Initialize database
    init_database()
    logger.info("Investment MCP server starting...")
    
    # Run the server using stdin/stdout streams
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            NotificationOptions(),
        )

if __name__ == "__main__":
    asyncio.run(main())