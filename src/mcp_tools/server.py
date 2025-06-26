"""
Unified MCP server for Investment MCP System.

This module consolidates all MCP functionality from various scattered
implementations into a single, efficient MCP server.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Sequence
from datetime import datetime, timedelta

from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from pydantic import AnyUrl

from ..collectors import MarketDataCollector, SwedishDataCollector
from ..analysis import RiskAnalyzer, TechnicalAnalyzer, PortfolioOptimizer
from ..ai import AIPortfolioAdvisor
from ..core.config import get_settings, TRADEABLE_FUNDS

logger = logging.getLogger(__name__)


class InvestmentMCPServer:
    """Unified MCP server for Investment analysis and portfolio management."""
    
    def __init__(self):
        self.settings = get_settings()
        self.server = Server("investment-mcp")
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        self.market_collector = MarketDataCollector()
        self.swedish_collector = SwedishDataCollector()
        self.risk_analyzer = RiskAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.ai_advisor = AIPortfolioAdvisor()
        
        # Register tools
        self._register_tools()
    
    def _register_tools(self):
        """Register all MCP tools."""
        
        # Data Collection Tools
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List all available tools."""
            return [
                Tool(
                    name="collect_fund_data",
                    description="Collect historical market data for approved Swedish funds",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "fund_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of fund IDs to collect (optional, collects all if not specified)"
                            },
                            "years": {
                                "type": "integer",
                                "description": "Number of years of historical data (default: 20)",
                                "minimum": 1,
                                "maximum": 25
                            }
                        }
                    }
                ),
                Tool(
                    name="collect_swedish_economic_data",
                    description="Collect Swedish economic indicators from Riksbanken and SCB",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "years": {
                                "type": "integer", 
                                "description": "Number of years of data (default: 5)",
                                "minimum": 1,
                                "maximum": 10
                            }
                        }
                    }
                ),
                Tool(
                    name="analyze_fund_risk",
                    description="Perform comprehensive risk analysis for a fund or portfolio",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "fund_id": {
                                "type": "string",
                                "description": "Single fund ID for analysis"
                            },
                            "portfolio": {
                                "type": "object",
                                "description": "Portfolio allocation (fund_id -> weight mapping)",
                                "additionalProperties": {"type": "number"}
                            }
                        },
                        "oneOf": [
                            {"required": ["fund_id"]},
                            {"required": ["portfolio"]}
                        ]
                    }
                ),
                Tool(
                    name="technical_analysis",
                    description="Perform technical analysis on fund price data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "fund_id": {
                                "type": "string",
                                "description": "Fund ID for technical analysis"
                            },
                            "indicators": {
                                "type": "array",
                                "items": {"type": "string", "enum": ["rsi", "macd", "bollinger", "moving_averages", "all"]},
                                "description": "Technical indicators to calculate (default: all)"
                            }
                        },
                        "required": ["fund_id"]
                    }
                ),
                Tool(
                    name="optimize_portfolio",
                    description="Optimize portfolio allocation using Modern Portfolio Theory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "risk_tolerance": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "very_high"],
                                "description": "Investor risk tolerance"
                            },
                            "optimization_method": {
                                "type": "string",
                                "enum": ["mean_variance", "risk_parity", "factor_based"],
                                "description": "Optimization method (default: mean_variance)"
                            },
                            "target_return": {
                                "type": "number",
                                "description": "Target annual return (optional)",
                                "minimum": 0,
                                "maximum": 0.5
                            }
                        },
                        "required": ["risk_tolerance"]
                    }
                ),
                Tool(
                    name="ai_portfolio_recommendation",
                    description="Generate AI-enhanced portfolio recommendation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "risk_tolerance": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "very_high"],
                                "description": "Investor risk tolerance"
                            },
                            "investment_horizon": {
                                "type": "string",
                                "enum": ["short", "medium", "long"],
                                "description": "Investment time horizon"
                            },
                            "investment_amount": {
                                "type": "number",
                                "description": "Total investment amount in SEK",
                                "minimum": 1000
                            },
                            "current_portfolio": {
                                "type": "object",
                                "description": "Current portfolio allocation (optional)",
                                "additionalProperties": {"type": "number"}
                            }
                        },
                        "required": ["risk_tolerance", "investment_horizon", "investment_amount"]
                    }
                ),
                Tool(
                    name="stress_test_portfolio",
                    description="Perform stress testing on portfolio under various scenarios",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "portfolio": {
                                "type": "object",
                                "description": "Portfolio allocation (fund_id -> weight mapping)",
                                "additionalProperties": {"type": "number"}
                            },
                            "scenarios": {
                                "type": "object",
                                "description": "Custom stress scenarios (optional)",
                                "additionalProperties": {
                                    "type": "object",
                                    "additionalProperties": {"type": "number"}
                                }
                            }
                        },
                        "required": ["portfolio"]
                    }
                ),
                Tool(
                    name="get_fund_universe",
                    description="Get information about all approved funds in the investment universe",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Filter by fund category (optional)"
                            }
                        }
                    }
                ),
                Tool(
                    name="market_sentiment_analysis",
                    description="Analyze current market sentiment using latest data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_ai_insights": {
                                "type": "boolean",
                                "description": "Include AI-generated market insights (default: true)"
                            }
                        }
                    }
                ),
                Tool(
                    name="explain_fund_selection",
                    description="Get AI explanation for specific fund selection reasoning",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "fund_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of fund IDs to explain"
                            }
                        },
                        "required": ["fund_ids"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            
            try:
                if name == "collect_fund_data":
                    return await self._handle_collect_fund_data(arguments)
                elif name == "collect_swedish_economic_data":
                    return await self._handle_collect_swedish_data(arguments)
                elif name == "analyze_fund_risk":
                    return await self._handle_analyze_risk(arguments)
                elif name == "technical_analysis":
                    return await self._handle_technical_analysis(arguments)
                elif name == "optimize_portfolio":
                    return await self._handle_optimize_portfolio(arguments)
                elif name == "ai_portfolio_recommendation":
                    return await self._handle_ai_portfolio_recommendation(arguments)
                elif name == "stress_test_portfolio":
                    return await self._handle_stress_test(arguments)
                elif name == "get_fund_universe":
                    return await self._handle_get_fund_universe(arguments)
                elif name == "market_sentiment_analysis":
                    return await self._handle_market_sentiment(arguments)
                elif name == "explain_fund_selection":
                    return await self._handle_explain_fund_selection(arguments)
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
                    
            except Exception as e:
                self.logger.error(f"Error calling tool {name}: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def _handle_collect_fund_data(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle fund data collection."""
        
        fund_ids = arguments.get("fund_ids")
        years = arguments.get("years", 20)
        
        self.logger.info(f"Collecting fund data: funds={fund_ids}, years={years}")
        
        try:
            data_dict = await self.market_collector.collect_historical_data(fund_ids, years)
            statistics = self.market_collector.get_fund_statistics(data_dict)
            
            result = {
                "status": "success",
                "collection_summary": {
                    "funds_requested": len(fund_ids) if fund_ids else "all",
                    "years_collected": years,
                    "successful_collections": len([s for s in statistics.values() if s.get("status") == "success"]),
                    "failed_collections": len([s for s in statistics.values() if s.get("status") != "success"])
                },
                "fund_statistics": statistics,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            error_result = {"status": "error", "error": str(e)}
            return [TextContent(type="text", text=json.dumps(error_result, indent=2))]
    
    async def _handle_collect_swedish_data(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle Swedish economic data collection."""
        
        years = arguments.get("years", 5)
        
        self.logger.info(f"Collecting Swedish economic data: years={years}")
        
        try:
            data_dict = await self.swedish_collector.collect_all_economic_data(years)
            
            result = {
                "status": "success",
                "collection_summary": {
                    "years_collected": years,
                    "data_sources": list(data_dict.keys()),
                    "total_series": len(data_dict),
                    "successful_series": len([k for k, v in data_dict.items() if not v.empty])
                },
                "data_overview": {k: len(v) for k, v in data_dict.items()},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            error_result = {"status": "error", "error": str(e)}
            return [TextContent(type="text", text=json.dumps(error_result, indent=2))]
    
    async def _handle_analyze_risk(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle risk analysis."""
        
        fund_id = arguments.get("fund_id")
        portfolio = arguments.get("portfolio")
        
        try:
            if fund_id:
                # Single fund risk analysis
                # In production, would fetch actual data
                mock_returns = self._get_mock_returns(fund_id)
                risk_metrics = self.risk_analyzer.calculate_basic_risk_metrics(mock_returns)
                
                result = {
                    "analysis_type": "single_fund",
                    "fund_id": fund_id,
                    "fund_name": TRADEABLE_FUNDS.get(fund_id, {}).get("name", fund_id),
                    "risk_metrics": risk_metrics,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            elif portfolio:
                # Portfolio risk analysis
                mock_returns_matrix = self._get_mock_returns_matrix(list(portfolio.keys()))
                portfolio_risk = self.risk_analyzer.calculate_portfolio_risk(mock_returns_matrix, portfolio)
                
                result = {
                    "analysis_type": "portfolio",
                    "portfolio_allocation": portfolio,
                    "portfolio_risk_metrics": portfolio_risk,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            error_result = {"status": "error", "error": str(e)}
            return [TextContent(type="text", text=json.dumps(error_result, indent=2))]
    
    async def _handle_technical_analysis(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle technical analysis."""
        
        fund_id = arguments["fund_id"]
        indicators = arguments.get("indicators", ["all"])
        
        try:
            # In production, would fetch actual market data
            mock_market_data = self._get_mock_market_data(fund_id)
            
            if "all" in indicators:
                technical_summary = self.technical_analyzer.generate_technical_summary(mock_market_data)
            else:
                # Calculate specific indicators
                technical_summary = {}
                prices = mock_market_data["close_price"]
                
                if "rsi" in indicators:
                    technical_summary["rsi"] = self.technical_analyzer.calculate_rsi(prices)
                if "macd" in indicators:
                    technical_summary["macd"] = self.technical_analyzer.calculate_macd(prices)
                if "bollinger" in indicators:
                    technical_summary["bollinger_bands"] = self.technical_analyzer.calculate_bollinger_bands(prices)
                if "moving_averages" in indicators:
                    technical_summary["moving_averages"] = self.technical_analyzer.calculate_moving_averages(prices)
            
            result = {
                "fund_id": fund_id,
                "fund_name": TRADEABLE_FUNDS.get(fund_id, {}).get("name", fund_id),
                "technical_analysis": technical_summary,
                "analysis_date": datetime.utcnow().isoformat()
            }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
            
        except Exception as e:
            error_result = {"status": "error", "error": str(e)}
            return [TextContent(type="text", text=json.dumps(error_result, indent=2))]
    
    async def _handle_optimize_portfolio(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle portfolio optimization."""
        
        risk_tolerance = arguments["risk_tolerance"]
        optimization_method = arguments.get("optimization_method", "mean_variance")
        target_return = arguments.get("target_return")
        
        try:
            # In production, would use real returns data
            mock_returns_matrix = self._get_mock_returns_matrix(list(TRADEABLE_FUNDS.keys())[:8])
            
            if optimization_method == "mean_variance":
                optimization_result = self.portfolio_optimizer.mean_variance_optimization(
                    mock_returns_matrix, target_return, risk_tolerance
                )
            elif optimization_method == "risk_parity":
                optimization_result = self.portfolio_optimizer.risk_parity_optimization(mock_returns_matrix)
            elif optimization_method == "factor_based":
                optimization_result = self.portfolio_optimizer.factor_based_optimization(mock_returns_matrix)
            else:
                raise ValueError(f"Unknown optimization method: {optimization_method}")
            
            result = {
                "optimization_request": {
                    "risk_tolerance": risk_tolerance,
                    "method": optimization_method,
                    "target_return": target_return
                },
                "optimization_result": optimization_result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            error_result = {"status": "error", "error": str(e)}
            return [TextContent(type="text", text=json.dumps(error_result, indent=2))]
    
    async def _handle_ai_portfolio_recommendation(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle AI portfolio recommendation."""
        
        risk_tolerance = arguments["risk_tolerance"]
        investment_horizon = arguments["investment_horizon"]
        investment_amount = arguments["investment_amount"]
        current_portfolio = arguments.get("current_portfolio")
        
        try:
            recommendation = await self.ai_advisor.generate_portfolio_recommendation(
                risk_tolerance=risk_tolerance,
                investment_horizon=investment_horizon,
                investment_amount=investment_amount,
                current_portfolio=current_portfolio
            )
            
            return [TextContent(type="text", text=json.dumps(recommendation, indent=2))]
            
        except Exception as e:
            error_result = {"status": "error", "error": str(e)}
            return [TextContent(type="text", text=json.dumps(error_result, indent=2))]
    
    async def _handle_stress_test(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle portfolio stress testing."""
        
        portfolio = arguments["portfolio"]
        custom_scenarios = arguments.get("scenarios")
        
        try:
            # In production, would use real returns data
            mock_returns_matrix = self._get_mock_returns_matrix(list(portfolio.keys()))
            
            stress_results = self.risk_analyzer.stress_test_portfolio(
                mock_returns_matrix, portfolio, custom_scenarios
            )
            
            result = {
                "portfolio_tested": portfolio,
                "stress_test_results": stress_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            error_result = {"status": "error", "error": str(e)}
            return [TextContent(type="text", text=json.dumps(error_result, indent=2))]
    
    async def _handle_get_fund_universe(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle fund universe information."""
        
        category = arguments.get("category")
        
        try:
            if category:
                # Filter by category
                filtered_funds = {
                    fund_id: fund_info 
                    for fund_id, fund_info in TRADEABLE_FUNDS.items()
                    if fund_info.get("category") == category
                }
                result = {
                    "category_filter": category,
                    "funds": filtered_funds,
                    "total_funds": len(filtered_funds)
                }
            else:
                # Return all funds
                result = {
                    "fund_universe": TRADEABLE_FUNDS,
                    "total_funds": len(TRADEABLE_FUNDS),
                    "categories": list(set(fund["category"] for fund in TRADEABLE_FUNDS.values())),
                    "risk_levels": list(set(fund["risk_level"] for fund in TRADEABLE_FUNDS.values()))
                }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            error_result = {"status": "error", "error": str(e)}
            return [TextContent(type="text", text=json.dumps(error_result, indent=2))]
    
    async def _handle_market_sentiment(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle market sentiment analysis."""
        
        include_ai_insights = arguments.get("include_ai_insights", True)
        
        try:
            # Mock market sentiment analysis
            sentiment_data = {
                "overall_sentiment": "cautiously_optimistic",
                "sentiment_score": 0.65,
                "key_factors": [
                    "Swedish economic growth outlook positive",
                    "Global market volatility elevated",
                    "Central bank policy supportive"
                ],
                "sector_sentiment": {
                    "technology": "positive",
                    "financials": "neutral",
                    "real_estate": "negative",
                    "commodities": "positive"
                }
            }
            
            if include_ai_insights:
                # In production, would call AI for market insights
                ai_insights = {
                    "ai_analysis": "Current market conditions favor diversified Swedish equity exposure with defensive positioning.",
                    "recommended_actions": [
                        "Maintain core Swedish equity allocation",
                        "Consider reducing real estate exposure",
                        "Monitor global market developments"
                    ]
                }
                sentiment_data["ai_insights"] = ai_insights
            
            result = {
                "market_sentiment_analysis": sentiment_data,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        except Exception as e:
            error_result = {"status": "error", "error": str(e)}
            return [TextContent(type="text", text=json.dumps(error_result, indent=2))]
    
    async def _handle_explain_fund_selection(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle fund selection explanation."""
        
        fund_ids = arguments["fund_ids"]
        
        try:
            explanation = await self.ai_advisor.explain_fund_selection(fund_ids)
            return [TextContent(type="text", text=json.dumps(explanation, indent=2))]
            
        except Exception as e:
            error_result = {"status": "error", "error": str(e)}
            return [TextContent(type="text", text=json.dumps(error_result, indent=2))]
    
    # Mock data methods (in production, these would fetch real data)
    def _get_mock_returns(self, fund_id: str):
        """Generate mock returns data for testing."""
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)  # For reproducible results
        dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="D")
        returns = np.random.normal(0.0008, 0.02, len(dates))  # Daily returns
        
        return pd.Series(returns, index=dates)
    
    def _get_mock_returns_matrix(self, fund_ids: List[str]):
        """Generate mock returns matrix for testing."""
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="D")
        
        returns_data = {}
        for i, fund_id in enumerate(fund_ids):
            # Different risk/return profiles for different funds
            mean_return = 0.0005 + i * 0.0001
            volatility = 0.015 + i * 0.002
            returns_data[fund_id] = np.random.normal(mean_return, volatility, len(dates))
        
        return pd.DataFrame(returns_data, index=dates)
    
    def _get_mock_market_data(self, fund_id: str):
        """Generate mock market data for testing."""
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")
        
        # Generate price series
        initial_price = 100
        returns = np.random.normal(0.0008, 0.02, len(dates))
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLC data
        data = {
            "date": dates,
            "close_price": prices,
            "open_price": [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            "high_price": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            "low_price": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            "volume": np.random.randint(10000, 100000, len(dates))
        }
        
        return pd.DataFrame(data)
    
    async def run(self):
        """Run the MCP server."""
        
        self.logger.info("Starting Investment MCP Server")
        
        # In production, would use proper transport
        # For now, this is a placeholder for the server run logic
        self.logger.info("Investment MCP Server is ready")
        
        # Keep server running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Shutting down Investment MCP Server")


# Factory function for easy server creation
def create_investment_mcp_server() -> InvestmentMCPServer:
    """Create and configure an Investment MCP server instance."""
    return InvestmentMCPServer()