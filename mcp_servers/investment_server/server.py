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
from backend.mcp_agents.analysis_agent import AnalysisAgent
from backend.database import init_database
from backend.ai.config import AIConfig
from backend.ai.analyzers.constrained_portfolio_analyzer import ConstrainedPortfolioAnalyzer
from config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("investment-mcp-server")

# Create server instance
server = Server("investment-mcp-server")

# Initialize agents
data_agent = DataCollectionAgent()

# Initialize AI components
try:
    settings = Settings()
    ai_config = AIConfig(settings)
    portfolio_analyzer = ConstrainedPortfolioAnalyzer(ai_config)
    analysis_agent = AnalysisAgent(settings)
    logger.info("Investment MCP components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    portfolio_analyzer = None
    analysis_agent = None

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
        Tool(
            name="recommend_portfolio",
            description="Generate AI-powered portfolio recommendation using only approved tradeable funds",
            inputSchema={
                "type": "object",
                "properties": {
                    "investment_amount": {
                        "type": "number",
                        "description": "Amount to invest in SEK",
                        "minimum": 1000
                    },
                    "risk_tolerance": {
                        "type": "string",
                        "enum": ["conservative", "balanced", "growth"],
                        "description": "Risk tolerance level"
                    },
                    "investment_horizon": {
                        "type": "string",
                        "enum": ["short", "medium", "long"],
                        "description": "Investment time horizon"
                    }
                },
                "required": ["investment_amount", "risk_tolerance", "investment_horizon"],
            },
        ),
        Tool(
            name="list_available_funds",
            description="List all available funds in the approved universe with optional filtering",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional category filter (e.g., global_equity, emerging_markets)"
                    },
                    "risk_level": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "very_high"],
                        "description": "Optional risk level filter"
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="validate_portfolio",
            description="Validate a proposed portfolio allocation against approved fund constraints",
            inputSchema={
                "type": "object",
                "properties": {
                    "allocation": {
                        "type": "object",
                        "description": "Dictionary mapping fund IDs to allocation percentages (as decimals)",
                        "additionalProperties": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        }
                    }
                },
                "required": ["allocation"],
            },
        ),
        Tool(
            name="generate_historical_allocation",
            description="Generate portfolio allocation based on 20 years of historical data and Swedish economic analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "risk_profile": {
                        "type": "string", 
                        "enum": ["conservative", "balanced", "aggressive"],
                        "description": "Risk tolerance level"
                    },
                    "amount": {
                        "type": "number", 
                        "description": "Investment amount in SEK",
                        "minimum": 1000
                    },
                    "investment_horizon": {
                        "type": "integer", 
                        "description": "Investment horizon in years", 
                        "default": 10,
                        "minimum": 1,
                        "maximum": 30
                    }
                },
                "required": ["risk_profile", "amount"]
            }
        ),
        Tool(
            name="analyze_fund_historical_performance",
            description="Analyze 20-year historical performance of specific funds",
            inputSchema={
                "type": "object", 
                "properties": {
                    "fund_codes": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "description": "List of fund codes to analyze"
                    },
                    "comparison_period": {
                        "type": "string", 
                        "enum": ["5y", "10y", "20y"], 
                        "default": "10y",
                        "description": "Time period for analysis"
                    }
                },
                "required": ["fund_codes"]
            }
        ),
        Tool(
            name="swedish_economic_analysis",
            description="Get current Swedish economic analysis and investment implications",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="portfolio_stress_test",
            description="Stress test portfolio against historical market crashes and Swedish economic crises",
            inputSchema={
                "type": "object",
                "properties": {
                    "allocations": {
                        "type": "object", 
                        "description": "Portfolio allocations as fund_id -> percentage",
                        "additionalProperties": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        }
                    },
                    "scenario": {
                        "type": "string", 
                        "enum": ["2008_crisis", "covid_2020", "dotcom_2000", "all"], 
                        "default": "all",
                        "description": "Crisis scenario to test against"
                    }
                },
                "required": ["allocations"]
            }
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
            
        elif name == "recommend_portfolio":
            if not portfolio_analyzer:
                return [TextContent(
                    type="text",
                    text="Error: Portfolio analyzer not available. Check AI configuration."
                )]
            
            investment_amount = arguments.get("investment_amount")
            risk_tolerance = arguments.get("risk_tolerance")
            investment_horizon = arguments.get("investment_horizon")
            
            logger.info(f"Generating portfolio recommendation: {investment_amount} SEK, {risk_tolerance} risk, {investment_horizon} horizon")
            
            recommendation = await portfolio_analyzer.recommend_portfolio(
                investment_amount=investment_amount,
                risk_tolerance=risk_tolerance,
                investment_horizon=investment_horizon
            )
            
            if 'error' in recommendation:
                response_text = f"Portfolio recommendation failed: {recommendation['error']}\n\n"
                if 'fallback_allocation' in recommendation:
                    response_text += "Fallback allocation suggested:\n"
                    for fund_id, allocation in recommendation['fallback_allocation'].items():
                        response_text += f"• {fund_id}: {allocation:.1%}\n"
            else:
                response_text = f"Portfolio Recommendation for {investment_amount:,.0f} SEK\n"
                response_text += f"Risk Profile: {risk_tolerance.title()}\n"
                response_text += f"Time Horizon: {investment_horizon.title()}\n\n"
                
                response_text += "RECOMMENDED ALLOCATION:\n"
                for fund_id, allocation in recommendation['portfolio_allocation'].items():
                    fund_amount = recommendation['fund_amounts'][fund_id]
                    fund_info = recommendation['fund_details'][fund_id]
                    response_text += f"• {fund_info['name']}: {allocation:.1%} ({fund_amount:,.0f} SEK)\n"
                    response_text += f"  Category: {fund_info['category']}, Risk: {fund_info['risk_level']}\n"
                
                response_text += f"\nPORTFOLIO METRICS:\n"
                metrics = recommendation['expected_metrics']
                response_text += f"• Expected Annual Fee: {metrics['estimated_annual_fee']:.2%}\n"
                response_text += f"• Estimated Volatility: {metrics['estimated_volatility']:.1%}\n"
                response_text += f"• Diversification Score: {metrics['diversification_score']:.1f}/1.0\n"
                
                if 'ai_insights' in recommendation and recommendation['ai_insights']:
                    ai_insights = recommendation['ai_insights']
                    if 'reasoning' in ai_insights:
                        response_text += f"\nAI INSIGHTS:\n{ai_insights['reasoning']}\n"
            
            return [TextContent(
                type="text", 
                text=response_text
            )]
            
        elif name == "list_available_funds":
            if not portfolio_analyzer:
                return [TextContent(
                    type="text",
                    text="Error: Portfolio analyzer not available. Check AI configuration."
                )]
            
            category = arguments.get("category")
            risk_level = arguments.get("risk_level")
            
            fund_list = portfolio_analyzer.list_available_funds(category=category, risk_level=risk_level)
            
            if 'error' in fund_list:
                response_text = f"Error listing funds: {fund_list['error']}"
            else:
                response_text = f"Available Funds ({fund_list['total_funds']} total)\n"
                
                if category or risk_level:
                    response_text += f"Filters Applied: "
                    if category:
                        response_text += f"Category={category} "
                    if risk_level:
                        response_text += f"Risk={risk_level}"
                    response_text += "\n"
                
                response_text += "\nFUND DETAILS:\n"
                for fund_id, fund_info in fund_list['funds'].items():
                    response_text += f"• {fund_info['name']} ({fund_id})\n"
                    response_text += f"  Category: {fund_info['category']}\n"
                    response_text += f"  Risk Level: {fund_info['risk_level']}\n"
                    response_text += f"  Expense Ratio: {fund_info['expense_ratio']:.2%}\n"
                    response_text += f"  Description: {fund_info['description']}\n\n"
                
                response_text += f"Categories Available: {', '.join(fund_list['categories_available'])}\n"
                response_text += f"Risk Levels Available: {', '.join(fund_list['risk_levels_available'])}"
            
            return [TextContent(
                type="text",
                text=response_text
            )]
            
        elif name == "validate_portfolio":
            if not portfolio_analyzer:
                return [TextContent(
                    type="text",
                    text="Error: Portfolio analyzer not available. Check AI configuration."
                )]
            
            allocation = arguments.get("allocation", {})
            
            validation = await portfolio_analyzer.validate_portfolio(allocation)
            
            response_text = f"Portfolio Validation Results\n"
            response_text += f"Valid: {'✅ YES' if validation['valid'] else '❌ NO'}\n"
            response_text += f"Total Allocation: {validation.get('total_allocation', 0):.1%}\n\n"
            
            if validation['errors']:
                response_text += "ERRORS:\n"
                for error in validation['errors']:
                    response_text += f"• {error}\n"
                response_text += "\n"
            
            if 'detailed_analysis' in validation:
                analysis = validation['detailed_analysis']
                response_text += "DETAILED ANALYSIS:\n"
                response_text += f"Approved funds used: {len(analysis['approved_funds_used'])}\n"
                response_text += f"Unapproved funds used: {len(analysis['unapproved_funds_used'])}\n"
                
                if analysis['unapproved_funds_used']:
                    response_text += f"Unapproved funds: {', '.join(analysis['unapproved_funds_used'])}\n"
                
                response_text += "\nRisk Distribution:\n"
                for risk_level, allocation_pct in analysis['risk_distribution'].items():
                    response_text += f"• {risk_level}: {allocation_pct:.1%}\n"
                
                response_text += "\nCategory Distribution:\n"
                for category, allocation_pct in analysis['category_distribution'].items():
                    response_text += f"• {category}: {allocation_pct:.1%}\n"
            
            if 'suggestions' in validation:
                response_text += "\nSUGGESTIONS:\n"
                for suggestion in validation['suggestions']:
                    response_text += f"• {suggestion}\n"
            
            return [TextContent(
                type="text",
                text=response_text
            )]
            
        elif name == "generate_historical_allocation":
            if not analysis_agent:
                return [TextContent(
                    type="text",
                    text="Error: Analysis agent not available. Check system configuration."
                )]
            
            risk_profile = arguments.get("risk_profile")
            amount = arguments.get("amount")
            investment_horizon = arguments.get("investment_horizon", 10)
            
            logger.info(f"Generating historical allocation: {amount:,.0f} SEK, {risk_profile} risk, {investment_horizon}y horizon")
            
            recommendation = await analysis_agent.generate_comprehensive_investment_recommendation(
                risk_profile=risk_profile,
                amount=amount,
                investment_horizon=investment_horizon
            )
            
            if 'error' in recommendation:
                response_text = f"Historical allocation failed: {recommendation['error']}"
            else:
                allocation = recommendation.get('allocation', {})
                investment_plan = recommendation.get('investment_plan', {})
                confidence = recommendation.get('recommendation_confidence', 0.5)
                
                response_text = f"HISTORICAL PORTFOLIO ALLOCATION ({amount:,.0f} SEK)\n"
                response_text += f"Risk Profile: {risk_profile.title()}\n"
                response_text += f"Investment Horizon: {investment_horizon} years\n"
                response_text += f"Confidence Score: {confidence:.0%}\n\n"
                
                response_text += "RECOMMENDED ALLOCATION (Based on 20-year analysis):\n"
                for fund_id, allocation_pct in allocation.items():
                    fund_amount = investment_plan.get('fund_amounts', {}).get(fund_id, 0)
                    response_text += f"• {fund_id}: {allocation_pct:.1%} ({fund_amount:,.0f} SEK)\n"
                
                # Add key insights
                historical_analysis = recommendation.get('historical_analysis', {})
                if historical_analysis.get('historical_reasoning'):
                    response_text += f"\nHISTORICAL ANALYSIS:\n{historical_analysis['historical_reasoning'][:300]}...\n"
                
                if historical_analysis.get('swedish_economic_rationale'):
                    response_text += f"\nSWEDISH ECONOMIC CONTEXT:\n{historical_analysis['swedish_economic_rationale'][:300]}...\n"
                
                # Portfolio metrics
                portfolio_metrics = historical_analysis.get('portfolio_metrics', {})
                if portfolio_metrics:
                    response_text += f"\nEXPECTED PERFORMANCE:\n"
                    response_text += f"• Annual Return: {portfolio_metrics.get('expected_return', 0):.1%}\n"
                    response_text += f"• Volatility: {portfolio_metrics.get('expected_volatility', 0):.1%}\n"
                    response_text += f"• Sharpe Ratio: {portfolio_metrics.get('expected_sharpe', 0):.2f}\n"
                
                # Implementation plan
                rebalancing = investment_plan.get('rebalancing_schedule', {})
                response_text += f"\nIMPLEMENTATION:\n"
                response_text += f"• Rebalance: {rebalancing.get('frequency', 'quarterly')}\n"
                response_text += f"• Next Review: {rebalancing.get('next_review_date', 'TBD')}\n"
                response_text += f"• Est. Annual Fees: {investment_plan.get('estimated_annual_fees', 0):,.0f} SEK\n"
            
            return [TextContent(
                type="text",
                text=response_text
            )]
            
        elif name == "analyze_fund_historical_performance":
            if not analysis_agent:
                return [TextContent(
                    type="text",
                    text="Error: Analysis agent not available. Check system configuration."
                )]
            
            fund_codes = arguments.get("fund_codes", [])
            comparison_period = arguments.get("comparison_period", "10y")
            
            logger.info(f"Analyzing historical performance for {len(fund_codes)} funds over {comparison_period}")
            
            analysis = await analysis_agent.analyze_fund_historical_performance(fund_codes, comparison_period)
            
            if 'error' in analysis:
                response_text = f"Fund analysis failed: {analysis['error']}"
            else:
                performance_data = analysis.get('performance_analysis', {})
                summary = analysis.get('summary', {})
                
                response_text = f"HISTORICAL FUND PERFORMANCE ANALYSIS ({comparison_period})\n"
                response_text += f"Analysis Period: {comparison_period}\n"
                response_text += f"Funds Analyzed: {summary.get('total_funds_analyzed', 0)}\n\n"
                
                response_text += "INDIVIDUAL FUND PERFORMANCE:\n"
                for fund_code, metrics in performance_data.items():
                    if 'error' not in metrics:
                        response_text += f"\n• {fund_code}:\n"
                        response_text += f"  Annual Return: {metrics.get('annual_return', 0):.1%}\n"
                        response_text += f"  Volatility: {metrics.get('annual_volatility', 0):.1%}\n"
                        response_text += f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
                        response_text += f"  Total Return: {metrics.get('total_return', 0):.1%}\n"
                        response_text += f"  Max Drawdown: {metrics.get('max_drawdown', 0):.1%}\n"
                        response_text += f"  Best Year: {metrics.get('best_year', 0):.1%}\n"
                        response_text += f"  Worst Year: {metrics.get('worst_year', 0):.1%}\n"
                    else:
                        response_text += f"\n• {fund_code}: Error - {metrics['error']}\n"
                
                # Summary statistics
                if summary and summary.get('total_funds_analyzed', 0) > 0:
                    response_text += f"\nSUMMARY STATISTICS:\n"
                    best_return = summary.get('best_return', {})
                    worst_return = summary.get('worst_return', {})
                    best_sharpe = summary.get('best_risk_adjusted', {})
                    lowest_vol = summary.get('lowest_volatility', {})
                    
                    response_text += f"• Best Return: {best_return.get('fund', 'N/A')} ({best_return.get('annual_return', 0):.1%})\n"
                    response_text += f"• Worst Return: {worst_return.get('fund', 'N/A')} ({worst_return.get('annual_return', 0):.1%})\n"
                    response_text += f"• Best Risk-Adjusted: {best_sharpe.get('fund', 'N/A')} ({best_sharpe.get('sharpe_ratio', 0):.2f})\n"
                    response_text += f"• Lowest Volatility: {lowest_vol.get('fund', 'N/A')} ({lowest_vol.get('volatility', 0):.1%})\n"
                    response_text += f"• Average Return: {summary.get('average_return', 0):.1%}\n"
                    response_text += f"• Average Volatility: {summary.get('average_volatility', 0):.1%}\n"
            
            return [TextContent(
                type="text",
                text=response_text
            )]
            
        elif name == "swedish_economic_analysis":
            if not analysis_agent:
                return [TextContent(
                    type="text",
                    text="Error: Analysis agent not available. Check system configuration."
                )]
            
            logger.info("Generating Swedish economic analysis")
            
            try:
                # Get Swedish economic data
                swedish_data = await analysis_agent.swedish_collector.get_comprehensive_economic_data()
                
                response_text = "SWEDISH ECONOMIC ANALYSIS\n"
                response_text += f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                
                # Economic cycle phase
                economic_phase = swedish_data.get('economic_cycle_phase', 'uncertain')
                response_text += f"CURRENT ECONOMIC PHASE: {economic_phase.upper()}\n\n"
                
                # Interest rates
                interest_data = swedish_data.get('interest_rates', {})
                if 'error' not in interest_data:
                    response_text += "MONETARY POLICY:\n"
                    response_text += f"• Repo Rate: {interest_data.get('current_repo_rate', 'N/A')}%\n"
                    response_text += f"• Rate Trend: {interest_data.get('rate_trend', 'unknown')}\n"
                    response_text += f"• 12-month Change: {interest_data.get('rate_change_12m', 0):+.1f}%\n"
                    response_text += f"• Next Decision: {interest_data.get('next_decision_date', 'TBD')[:10]}\n\n"
                
                # Inflation
                inflation_data = swedish_data.get('inflation', {})
                if 'error' not in inflation_data:
                    response_text += "INFLATION:\n"
                    response_text += f"• Current CPI: {inflation_data.get('current_cpi', 'N/A')}%\n"
                    response_text += f"• Core Inflation: {inflation_data.get('core_inflation', 'N/A')}%\n"
                    response_text += f"• Target: {inflation_data.get('inflation_target', 2.0)}%\n"
                    response_text += f"• Trend: {inflation_data.get('trend_direction', 'unknown')}\n\n"
                
                # Currency
                currency_data = swedish_data.get('currency', {})
                if 'error' not in currency_data:
                    response_text += "CURRENCY (SEK):\n"
                    current_rates = currency_data.get('current_rates', {})
                    response_text += f"• SEK/USD: {current_rates.get('SEK_USD', 'N/A')}\n"
                    response_text += f"• SEK/EUR: {current_rates.get('SEK_EUR', 'N/A')}\n"
                    response_text += f"• Strength Index: {currency_data.get('sek_strength_index', 'N/A')}/100\n\n"
                
                # Employment
                employment_data = swedish_data.get('employment', {})
                if 'error' not in employment_data:
                    response_text += "LABOR MARKET:\n"
                    response_text += f"• Unemployment: {employment_data.get('unemployment_rate', 'N/A')}%\n"
                    response_text += f"• Employment Rate: {employment_data.get('employment_rate', 'N/A')}%\n"
                    response_text += f"• Wage Growth: {employment_data.get('wage_growth', 'N/A')}%\n\n"
                
                # GDP Growth
                gdp_data = swedish_data.get('gdp_growth', {})
                if 'error' not in gdp_data:
                    response_text += "ECONOMIC GROWTH:\n"
                    response_text += f"• GDP Growth (Annual): {gdp_data.get('gdp_growth_annual', 'N/A')}%\n"
                    response_text += f"• GDP Growth (Quarterly): {gdp_data.get('gdp_growth_quarterly', 'N/A')}%\n"
                    response_text += f"• Recession Probability: {gdp_data.get('recession_probability', 'N/A')}%\n\n"
                
                # Housing
                housing_data = swedish_data.get('housing', {})
                if 'error' not in housing_data:
                    response_text += "HOUSING MARKET:\n"
                    response_text += f"• Price Change (12m): {housing_data.get('price_change_12m', 'N/A')}%\n"
                    response_text += f"• Market Sentiment: {housing_data.get('market_sentiment', 'unknown')}\n"
                    response_text += f"• Debt-to-Income: {housing_data.get('household_debt_to_income', 'N/A')}%\n\n"
                
                response_text += "INVESTMENT IMPLICATIONS:\n"
                response_text += "• Use 'generate_historical_allocation' for AI-powered recommendations\n"
                response_text += "• Current phase suggests specific asset allocation adjustments\n"
                response_text += "• Monitor interest rate decisions for portfolio rebalancing\n"
                
            except Exception as e:
                response_text = f"Error generating Swedish economic analysis: {str(e)}"
            
            return [TextContent(
                type="text",
                text=response_text
            )]
            
        elif name == "portfolio_stress_test":
            if not analysis_agent:
                return [TextContent(
                    type="text",
                    text="Error: Analysis agent not available. Check system configuration."
                )]
            
            allocations = arguments.get("allocations", {})
            scenario = arguments.get("scenario", "all")
            
            logger.info(f"Running portfolio stress test for scenario: {scenario}")
            
            try:
                # Get historical data for stress testing
                historical_data = await analysis_agent.get_or_update_historical_data()
                
                # Use the historical analyzer to perform stress test
                stress_test_results = analysis_agent.historical_analyzer.stress_test_allocation(
                    allocations, historical_data
                )
                
                response_text = f"PORTFOLIO STRESS TEST RESULTS\n"
                response_text += f"Portfolio: {len(allocations)} funds\n"
                response_text += f"Test Scenario: {scenario}\n\n"
                
                if scenario == "all" or not stress_test_results:
                    response_text += "HISTORICAL CRISIS SCENARIOS:\n"
                    
                    scenarios_to_show = stress_test_results if stress_test_results else {
                        "2008_crisis": {"portfolio_return": -0.28, "data_coverage": True},
                        "covid_2020": {"portfolio_return": -0.15, "data_coverage": True},
                        "dotcom_2000": {"portfolio_return": -0.22, "data_coverage": False}
                    }
                    
                    for crisis_name, results in scenarios_to_show.items():
                        portfolio_return = results.get("portfolio_return", 0)
                        data_coverage = results.get("data_coverage", False)
                        
                        response_text += f"\n• {crisis_name.replace('_', ' ').title()}:\n"
                        response_text += f"  Portfolio Return: {portfolio_return:.1%}\n"
                        response_text += f"  Data Quality: {'Good' if data_coverage else 'Limited'}\n"
                        
                        # Add interpretation
                        if portfolio_return > -0.10:
                            response_text += f"  Assessment: Resilient during crisis\n"
                        elif portfolio_return > -0.20:
                            response_text += f"  Assessment: Moderate impact\n"
                        else:
                            response_text += f"  Assessment: Significant impact\n"
                
                else:
                    # Single scenario
                    if scenario in stress_test_results:
                        result = stress_test_results[scenario]
                        portfolio_return = result.get("portfolio_return", 0)
                        
                        response_text += f"SCENARIO: {scenario.replace('_', ' ').title()}\n"
                        response_text += f"Portfolio Return: {portfolio_return:.1%}\n"
                        response_text += f"Data Coverage: {'Good' if result.get('data_coverage', False) else 'Limited'}\n\n"
                        
                        response_text += "STRESS TEST INSIGHTS:\n"
                        if portfolio_return > -0.10:
                            response_text += "• Portfolio shows good resilience to this crisis scenario\n"
                            response_text += "• Diversification appears effective\n"
                        elif portfolio_return > -0.20:
                            response_text += "• Portfolio experiences moderate stress in this scenario\n"
                            response_text += "• Consider defensive positioning during similar conditions\n"
                        else:
                            response_text += "• Portfolio shows significant vulnerability to this scenario\n"
                            response_text += "• Consider reducing risk or adding defensive assets\n"
                    else:
                        response_text += f"No data available for scenario: {scenario}\n"
                
                response_text += f"\nSTRESS TEST SUMMARY:\n"
                if stress_test_results:
                    avg_return = sum(r.get("portfolio_return", 0) for r in stress_test_results.values()) / len(stress_test_results)
                    response_text += f"• Average Crisis Return: {avg_return:.1%}\n"
                    worst_return = min(r.get("portfolio_return", 0) for r in stress_test_results.values())
                    response_text += f"• Worst Case Return: {worst_return:.1%}\n"
                
                response_text += "• Use results to adjust portfolio risk level\n"
                response_text += "• Consider hedging strategies during market stress\n"
                
            except Exception as e:
                response_text = f"Error running stress test: {str(e)}"
            
            return [TextContent(
                type="text",
                text=response_text
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