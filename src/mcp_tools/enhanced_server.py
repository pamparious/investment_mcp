"""
Enhanced MCP server for Investment MCP System Phase 4.

This module provides the complete MCP server with AI-powered portfolio optimization,
Monte Carlo simulations, backtesting, and Swedish housing market analysis.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Sequence
from datetime import datetime, timedelta

from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from pydantic import AnyUrl

from ..ai.ai_engine import AIEngine
from ..core.optimization.portfolio_engine import PortfolioOptimizationEngine
from ..core.optimization.monte_carlo import MonteCarloSimulator
from ..core.optimization.backtesting import PortfolioBacktester
from ..core.optimization.housing_analysis import SwedishHousingAnalyzer
from ..core.data_collection.yfinance_collector import YFinanceCollector
from ..core.data_collection.riksbank_collector import RiksbankCollector
from ..core.config import TRADEABLE_FUNDS
from .tools import MCPToolValidator, MCPResponseFormatter

logger = logging.getLogger(__name__)


class EnhancedInvestmentMCPServer:
    """Enhanced MCP server with Phase 4 AI and optimization capabilities."""
    
    def __init__(self):
        self.server = Server("investment-mcp-enhanced")
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize Phase 4 components
        self.ai_engine = AIEngine()
        self.portfolio_engine = PortfolioOptimizationEngine()
        self.monte_carlo = MonteCarloSimulator()
        self.backtester = PortfolioBacktester()
        self.housing_analyzer = SwedishHousingAnalyzer()
        
        # Initialize data collectors
        self.yfinance_collector = YFinanceCollector()
        self.riksbank_collector = RiksbankCollector()
        
        # Utility components
        self.validator = MCPToolValidator()
        self.formatter = MCPResponseFormatter()
        
        # Server initialization status
        self.initialization_complete = False
        
        # Register enhanced tools
        self._register_enhanced_tools()
    
    async def initialize_server(self) -> bool:
        """Initialize server and all AI components."""
        
        try:
            self.logger.info("Initializing Enhanced Investment MCP Server")
            
            # Initialize AI engine
            ai_initialized = await self.ai_engine.initialize()
            if not ai_initialized:
                self.logger.warning("AI engine initialization failed - some features may be limited")
            
            self.initialization_complete = True
            self.logger.info("Enhanced Investment MCP Server initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Server initialization failed: {e}")
            return False
    
    def _register_enhanced_tools(self):
        """Register all enhanced MCP tools."""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List all available enhanced tools."""
            return [
                # Phase 4 AI-Powered Tools
                Tool(
                    name="ai_portfolio_analysis",
                    description="AI-powered comprehensive portfolio analysis using Gemma 3:1B",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "allocation": {
                                "type": "object",
                                "description": "Portfolio allocation (fund_id -> weight)",
                                "additionalProperties": {"type": "number"}
                            },
                            "risk_tolerance": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "very_high"],
                                "description": "Risk tolerance level"
                            },
                            "investment_amount": {
                                "type": "number",
                                "description": "Investment amount in SEK",
                                "minimum": 1000
                            },
                            "market_context": {
                                "type": "object",
                                "description": "Additional market context (optional)",
                                "additionalProperties": True
                            }
                        },
                        "required": ["allocation", "risk_tolerance", "investment_amount"]
                    }
                ),
                Tool(
                    name="optimize_portfolio_ai",
                    description="AI-enhanced portfolio optimization with multiple algorithms",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "optimization_type": {
                                "type": "string",
                                "enum": ["markowitz", "risk_parity", "maximum_sharpe", "minimum_volatility", "equal_weight"],
                                "description": "Optimization algorithm to use"
                            },
                            "risk_tolerance": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "very_high"],
                                "description": "Risk tolerance level"
                            },
                            "target_return": {
                                "type": "number",
                                "description": "Target annual return (optional)",
                                "minimum": 0,
                                "maximum": 1
                            },
                            "target_volatility": {
                                "type": "number",
                                "description": "Target volatility (optional)",
                                "minimum": 0,
                                "maximum": 1
                            },
                            "current_portfolio": {
                                "type": "object",
                                "description": "Current portfolio for rebalancing (optional)",
                                "additionalProperties": {"type": "number"}
                            }
                        },
                        "required": ["optimization_type", "risk_tolerance"]
                    }
                ),
                Tool(
                    name="monte_carlo_analysis",
                    description="Monte Carlo simulation for portfolio risk analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "allocation": {
                                "type": "object",
                                "description": "Portfolio allocation",
                                "additionalProperties": {"type": "number"}
                            },
                            "initial_investment": {
                                "type": "number",
                                "description": "Initial investment amount in SEK",
                                "minimum": 1000,
                                "default": 1000000
                            },
                            "time_horizon_years": {
                                "type": "number",
                                "description": "Investment time horizon in years",
                                "minimum": 1,
                                "maximum": 30,
                                "default": 10
                            },
                            "n_simulations": {
                                "type": "integer",
                                "description": "Number of Monte Carlo simulations",
                                "minimum": 1000,
                                "maximum": 50000,
                                "default": 10000
                            },
                            "include_rebalancing": {
                                "type": "boolean",
                                "description": "Include periodic rebalancing",
                                "default": false
                            }
                        },
                        "required": ["allocation"]
                    }
                ),
                Tool(
                    name="portfolio_stress_test",
                    description="Comprehensive stress testing with Swedish market scenarios",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "allocation": {
                                "type": "object",
                                "description": "Portfolio allocation",
                                "additionalProperties": {"type": "number"}
                            },
                            "custom_scenarios": {
                                "type": "array",
                                "description": "Custom stress scenarios (optional)",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "description": {"type": "string"},
                                        "return_shocks": {"type": "object"},
                                        "volatility_multipliers": {"type": "object"}
                                    }
                                }
                            },
                            "n_simulations": {
                                "type": "integer",
                                "description": "Simulations per scenario",
                                "minimum": 1000,
                                "default": 5000
                            }
                        },
                        "required": ["allocation"]
                    }
                ),
                Tool(
                    name="portfolio_backtest",
                    description="Historical backtesting of portfolio strategies",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "strategy_type": {
                                "type": "string",
                                "enum": ["buy_and_hold", "rebalanced", "tactical"],
                                "description": "Backtesting strategy type"
                            },
                            "allocation": {
                                "type": "object",
                                "description": "Portfolio allocation",
                                "additionalProperties": {"type": "number"}
                            },
                            "start_date": {
                                "type": "string",
                                "description": "Backtest start date (YYYY-MM-DD)"
                            },
                            "end_date": {
                                "type": "string",
                                "description": "Backtest end date (YYYY-MM-DD)"
                            },
                            "initial_capital": {
                                "type": "number",
                                "description": "Initial capital in SEK",
                                "default": 1000000
                            },
                            "rebalancing_frequency": {
                                "type": "string",
                                "enum": ["monthly", "quarterly", "annually"],
                                "description": "Rebalancing frequency",
                                "default": "quarterly"
                            }
                        },
                        "required": ["strategy_type", "allocation", "start_date", "end_date"]
                    }
                ),
                Tool(
                    name="housing_vs_investment",
                    description="Swedish housing market vs portfolio investment analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "property_price": {
                                "type": "number",
                                "description": "Property price in SEK",
                                "minimum": 100000
                            },
                            "monthly_rent": {
                                "type": "number",
                                "description": "Monthly rent savings in SEK",
                                "minimum": 1000
                            },
                            "portfolio_allocation": {
                                "type": "object",
                                "description": "Alternative portfolio allocation",
                                "additionalProperties": {"type": "number"}
                            },
                            "expected_portfolio_return": {
                                "type": "number",
                                "description": "Expected annual portfolio return",
                                "minimum": 0,
                                "maximum": 1
                            },
                            "portfolio_volatility": {
                                "type": "number",
                                "description": "Portfolio volatility",
                                "minimum": 0,
                                "maximum": 1
                            },
                            "analysis_period_years": {
                                "type": "integer",
                                "description": "Analysis period in years",
                                "minimum": 1,
                                "maximum": 30,
                                "default": 10
                            },
                            "region": {
                                "type": "string",
                                "enum": ["Stockholm", "Göteborg", "Malmö"],
                                "description": "Swedish region",
                                "default": "Stockholm"
                            }
                        },
                        "required": ["property_price", "monthly_rent", "portfolio_allocation", "expected_portfolio_return", "portfolio_volatility"]
                    }
                ),
                Tool(
                    name="mortgage_affordability",
                    description="Calculate mortgage affordability under Swedish regulations",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "annual_income": {
                                "type": "number",
                                "description": "Annual income in SEK",
                                "minimum": 100000
                            },
                            "monthly_expenses": {
                                "type": "number",
                                "description": "Monthly expenses in SEK",
                                "minimum": 0
                            },
                            "existing_debt": {
                                "type": "number",
                                "description": "Existing monthly debt payments in SEK",
                                "minimum": 0,
                                "default": 0
                            },
                            "down_payment_available": {
                                "type": "number",
                                "description": "Available down payment in SEK",
                                "minimum": 0,
                                "default": 0
                            }
                        },
                        "required": ["annual_income", "monthly_expenses"]
                    }
                ),
                Tool(
                    name="compare_optimization_strategies",
                    description="Compare multiple portfolio optimization strategies",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "risk_tolerance": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "very_high"],
                                "description": "Risk tolerance level"
                            },
                            "strategies": {
                                "type": "array",
                                "description": "Strategies to compare (optional)",
                                "items": {
                                    "type": "string",
                                    "enum": ["markowitz", "risk_parity", "maximum_sharpe", "minimum_volatility", "equal_weight"]
                                }
                            }
                        },
                        "required": ["risk_tolerance"]
                    }
                ),
                Tool(
                    name="ai_market_commentary",
                    description="AI-generated Swedish market commentary and insights",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_economic_data": {
                                "type": "boolean",
                                "description": "Include Swedish economic data",
                                "default": true
                            },
                            "include_fund_performance": {
                                "type": "boolean",
                                "description": "Include fund performance analysis",
                                "default": true
                            }
                        }
                    }
                ),
                Tool(
                    name="efficient_frontier_analysis",
                    description="Calculate and analyze efficient frontier",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "risk_tolerance": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "very_high"],
                                "description": "Risk tolerance level"
                            },
                            "n_points": {
                                "type": "integer",
                                "description": "Number of frontier points",
                                "minimum": 10,
                                "maximum": 100,
                                "default": 25
                            }
                        },
                        "required": ["risk_tolerance"]
                    }
                ),
                Tool(
                    name="portfolio_rebalancing_advice",
                    description="AI-powered portfolio rebalancing recommendations",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "current_portfolio": {
                                "type": "object",
                                "description": "Current portfolio allocation",
                                "additionalProperties": {"type": "number"}
                            },
                            "target_portfolio": {
                                "type": "object",
                                "description": "Target portfolio allocation",
                                "additionalProperties": {"type": "number"}
                            },
                            "market_conditions": {
                                "type": "string",
                                "enum": ["bull", "bear", "volatile", "normal"],
                                "description": "Current market conditions",
                                "default": "normal"
                            }
                        },
                        "required": ["current_portfolio", "target_portfolio"]
                    }
                ),
                Tool(
                    name="var_cvar_analysis",
                    description="Value at Risk and Conditional VaR analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "allocation": {
                                "type": "object",
                                "description": "Portfolio allocation",
                                "additionalProperties": {"type": "number"}
                            },
                            "confidence_level": {
                                "type": "number",
                                "description": "Confidence level for VaR",
                                "minimum": 0.01,
                                "maximum": 0.25,
                                "default": 0.05
                            },
                            "time_horizon_days": {
                                "type": "integer",
                                "description": "Time horizon in days",
                                "minimum": 1,
                                "maximum": 252,
                                "default": 22
                            },
                            "initial_investment": {
                                "type": "number",
                                "description": "Initial investment amount",
                                "default": 1000000
                            }
                        },
                        "required": ["allocation"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle enhanced tool calls."""
            
            try:
                if not self.initialization_complete:
                    await self.initialize_server()
                
                # Phase 4 AI-Powered Tools
                if name == "ai_portfolio_analysis":
                    return await self._handle_ai_portfolio_analysis(arguments)
                elif name == "optimize_portfolio_ai":
                    return await self._handle_optimize_portfolio_ai(arguments)
                elif name == "monte_carlo_analysis":
                    return await self._handle_monte_carlo_analysis(arguments)
                elif name == "portfolio_stress_test":
                    return await self._handle_portfolio_stress_test(arguments)
                elif name == "portfolio_backtest":
                    return await self._handle_portfolio_backtest(arguments)
                elif name == "housing_vs_investment":
                    return await self._handle_housing_vs_investment(arguments)
                elif name == "mortgage_affordability":
                    return await self._handle_mortgage_affordability(arguments)
                elif name == "compare_optimization_strategies":
                    return await self._handle_compare_optimization_strategies(arguments)
                elif name == "ai_market_commentary":
                    return await self._handle_ai_market_commentary(arguments)
                elif name == "efficient_frontier_analysis":
                    return await self._handle_efficient_frontier_analysis(arguments)
                elif name == "portfolio_rebalancing_advice":
                    return await self._handle_portfolio_rebalancing_advice(arguments)
                elif name == "var_cvar_analysis":
                    return await self._handle_var_cvar_analysis(arguments)
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
                    
            except Exception as e:
                self.logger.error(f"Error calling tool {name}: {e}")
                error_response = self.formatter.error_response(str(e))
                return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    
    # Tool Handler Methods
    
    async def _handle_ai_portfolio_analysis(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle AI-powered portfolio analysis."""
        
        allocation = arguments["allocation"]
        risk_tolerance = arguments["risk_tolerance"]
        investment_amount = arguments["investment_amount"]
        market_context = arguments.get("market_context")
        
        # Validate allocation
        validation = self.validator.validate_portfolio_allocation(allocation)
        if not validation["valid"]:
            response = self.formatter.validation_error_response(validation)
            return [TextContent(type="text", text=json.dumps(response, indent=2))]
        
        try:
            # Generate AI analysis
            ai_analysis = await self.ai_engine.generate_portfolio_recommendation(
                current_allocation=allocation,
                risk_tolerance=risk_tolerance,
                investment_amount=investment_amount,
                market_context=market_context
            )
            
            # Get quantitative metrics (using mock data for now)
            mock_data = await self._get_portfolio_data(allocation)
            quantitative_metrics = await self._calculate_portfolio_metrics(allocation, mock_data)
            
            result = {
                "ai_analysis": ai_analysis,
                "quantitative_metrics": quantitative_metrics,
                "portfolio_summary": self._get_portfolio_summary(allocation),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            response = self.formatter.success_response(result, "AI portfolio analysis completed")
            return [TextContent(type="text", text=json.dumps(response, indent=2))]
            
        except Exception as e:
            error_response = self.formatter.error_response(f"AI analysis failed: {str(e)}")
            return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    
    async def _handle_optimize_portfolio_ai(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle AI-enhanced portfolio optimization."""
        
        optimization_type = arguments["optimization_type"]
        risk_tolerance = arguments["risk_tolerance"]
        target_return = arguments.get("target_return")
        target_volatility = arguments.get("target_volatility")
        current_portfolio = arguments.get("current_portfolio")
        
        try:
            # Get historical data (mock for now)
            expected_returns, covariance_matrix = await self._get_market_data()
            
            # Run optimization
            optimization_result = self.portfolio_engine.optimize_portfolio(
                optimization_type=optimization_type,
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                risk_tolerance=risk_tolerance,
                target_return=target_return,
                target_volatility=target_volatility,
                current_portfolio=current_portfolio
            )
            
            # Generate AI insights about the optimization
            if optimization_result["success"]:
                ai_insights = await self.ai_engine.generate_portfolio_recommendation(
                    current_allocation=optimization_result["allocation"],
                    risk_tolerance=risk_tolerance,
                    investment_amount=1000000
                )
                optimization_result["ai_insights"] = ai_insights
            
            response = self.formatter.success_response(optimization_result, "Portfolio optimization completed")
            return [TextContent(type="text", text=json.dumps(response, indent=2))]
            
        except Exception as e:
            error_response = self.formatter.error_response(f"Optimization failed: {str(e)}")
            return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    
    async def _handle_monte_carlo_analysis(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle Monte Carlo simulation analysis."""
        
        allocation = arguments["allocation"]
        initial_investment = arguments.get("initial_investment", 1000000)
        time_horizon_years = arguments.get("time_horizon_years", 10)
        n_simulations = arguments.get("n_simulations", 10000)
        include_rebalancing = arguments.get("include_rebalancing", False)
        
        # Validate allocation
        validation = self.validator.validate_portfolio_allocation(allocation)
        if not validation["valid"]:
            response = self.formatter.validation_error_response(validation)
            return [TextContent(type="text", text=json.dumps(response, indent=2))]
        
        try:
            # Get market data
            expected_returns, covariance_matrix = await self._get_market_data()
            
            # Run Monte Carlo simulation
            simulation_result = self.monte_carlo.run_portfolio_simulation(
                allocation=allocation,
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                initial_investment=initial_investment,
                time_horizon_years=time_horizon_years,
                n_simulations=n_simulations,
                include_rebalancing=include_rebalancing
            )
            
            response = self.formatter.success_response(simulation_result, "Monte Carlo analysis completed")
            return [TextContent(type="text", text=json.dumps(response, indent=2))]
            
        except Exception as e:
            error_response = self.formatter.error_response(f"Monte Carlo analysis failed: {str(e)}")
            return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    
    async def _handle_portfolio_stress_test(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle portfolio stress testing."""
        
        allocation = arguments["allocation"]
        custom_scenarios = arguments.get("custom_scenarios")
        n_simulations = arguments.get("n_simulations", 5000)
        
        try:
            # Get market data
            expected_returns, covariance_matrix = await self._get_market_data()
            
            # Generate stress scenarios if not provided
            if not custom_scenarios:
                stress_scenarios = self._generate_default_stress_scenarios()
            else:
                stress_scenarios = custom_scenarios
            
            # Run stress test
            stress_result = self.monte_carlo.run_stress_test(
                allocation=allocation,
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                stress_scenarios=stress_scenarios,
                n_simulations=n_simulations
            )
            
            response = self.formatter.success_response(stress_result, "Stress test completed")
            return [TextContent(type="text", text=json.dumps(response, indent=2))]
            
        except Exception as e:
            error_response = self.formatter.error_response(f"Stress test failed: {str(e)}")
            return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    
    async def _handle_housing_vs_investment(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle housing vs investment analysis."""
        
        property_price = arguments["property_price"]
        monthly_rent = arguments["monthly_rent"]
        portfolio_allocation = arguments["portfolio_allocation"]
        expected_portfolio_return = arguments["expected_portfolio_return"]
        portfolio_volatility = arguments["portfolio_volatility"]
        analysis_period_years = arguments.get("analysis_period_years", 10)
        region = arguments.get("region", "Stockholm")
        
        try:
            # Run housing vs investment analysis
            analysis_result = self.housing_analyzer.analyze_housing_vs_investment(
                property_price=property_price,
                monthly_rent_savings=monthly_rent,
                portfolio_allocation=portfolio_allocation,
                expected_portfolio_return=expected_portfolio_return,
                portfolio_volatility=portfolio_volatility,
                analysis_period_years=analysis_period_years,
                region=region
            )
            
            # Generate AI insights
            housing_ai_insights = await self.ai_engine.analyze_housing_vs_investment(
                housing_data={
                    "property_price": property_price,
                    "monthly_rent": monthly_rent,
                    "region": region
                },
                investment_returns={"portfolio": expected_portfolio_return},
                personal_situation={
                    "analysis_period": analysis_period_years,
                    "investment_amount": property_price * 0.15  # Assume 15% down payment
                }
            )
            
            analysis_result["ai_insights"] = housing_ai_insights
            
            response = self.formatter.success_response(analysis_result, "Housing vs investment analysis completed")
            return [TextContent(type="text", text=json.dumps(response, indent=2))]
            
        except Exception as e:
            error_response = self.formatter.error_response(f"Housing analysis failed: {str(e)}")
            return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    
    async def _handle_mortgage_affordability(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle mortgage affordability calculation."""
        
        annual_income = arguments["annual_income"]
        monthly_expenses = arguments["monthly_expenses"]
        existing_debt = arguments.get("existing_debt", 0)
        down_payment_available = arguments.get("down_payment_available", 0)
        
        try:
            # Calculate mortgage affordability
            affordability_result = self.housing_analyzer.calculate_mortgage_affordability(
                annual_income=annual_income,
                monthly_expenses=monthly_expenses,
                existing_debt=existing_debt,
                down_payment_available=down_payment_available
            )
            
            response = self.formatter.success_response(affordability_result, "Mortgage affordability calculated")
            return [TextContent(type="text", text=json.dumps(response, indent=2))]
            
        except Exception as e:
            error_response = self.formatter.error_response(f"Affordability calculation failed: {str(e)}")
            return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    
    async def _handle_compare_optimization_strategies(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle optimization strategy comparison."""
        
        risk_tolerance = arguments["risk_tolerance"]
        strategies = arguments.get("strategies", ["markowitz", "risk_parity", "maximum_sharpe", "minimum_volatility", "equal_weight"])
        
        try:
            # Get market data
            expected_returns, covariance_matrix = await self._get_market_data()
            
            # Compare strategies
            comparison_result = self.portfolio_engine.compare_optimization_strategies(
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                risk_tolerance=risk_tolerance,
                strategies_to_compare=strategies
            )
            
            response = self.formatter.success_response(comparison_result, "Strategy comparison completed")
            return [TextContent(type="text", text=json.dumps(response, indent=2))]
            
        except Exception as e:
            error_response = self.formatter.error_response(f"Strategy comparison failed: {str(e)}")
            return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    
    async def _handle_ai_market_commentary(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle AI market commentary generation."""
        
        include_economic_data = arguments.get("include_economic_data", True)
        include_fund_performance = arguments.get("include_fund_performance", True)
        
        try:
            # Collect market data
            swedish_economic_data = {}
            fund_performance = {}
            
            if include_economic_data:
                # Mock Swedish economic data
                swedish_economic_data = {
                    "riksbank_rate": 2.5,
                    "inflation_rate": 3.2,
                    "unemployment_rate": 7.1,
                    "gdp_growth": 1.8
                }
            
            if include_fund_performance:
                # Mock fund performance data
                fund_performance = {
                    "PLUS_ALLABOLAG_SVERIGE_INDEX": {"return": 0.12},
                    "DNB_GLOBAL_INDEKS_S": {"return": 0.15},
                    "AVANZA_USA": {"return": 0.18}
                }
            
            # Generate AI commentary
            commentary_result = await self.ai_engine.generate_market_commentary(
                swedish_economic_data=swedish_economic_data,
                fund_performance=fund_performance
            )
            
            response = self.formatter.success_response(commentary_result, "Market commentary generated")
            return [TextContent(type="text", text=json.dumps(response, indent=2))]
            
        except Exception as e:
            error_response = self.formatter.error_response(f"Market commentary failed: {str(e)}")
            return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    
    async def _handle_var_cvar_analysis(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle VaR and CVaR analysis."""
        
        allocation = arguments["allocation"]
        confidence_level = arguments.get("confidence_level", 0.05)
        time_horizon_days = arguments.get("time_horizon_days", 22)
        initial_investment = arguments.get("initial_investment", 1000000)
        
        try:
            # Get market data
            expected_returns, covariance_matrix = await self._get_market_data()
            
            # Calculate VaR and CVaR
            var_result = self.monte_carlo.calculate_var_and_cvar(
                allocation=allocation,
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                confidence_level=confidence_level,
                time_horizon_days=time_horizon_days,
                initial_investment=initial_investment
            )
            
            response = self.formatter.success_response(var_result, "VaR/CVaR analysis completed")
            return [TextContent(type="text", text=json.dumps(response, indent=2))]
            
        except Exception as e:
            error_response = self.formatter.error_response(f"VaR/CVaR analysis failed: {str(e)}")
            return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    
    # Helper methods (placeholder implementations)
    
    async def _get_market_data(self):
        """Get market data for optimization (placeholder)."""
        import pandas as pd
        import numpy as np
        
        # Mock data generation
        funds = list(TRADEABLE_FUNDS.keys())[:8]  # Use first 8 funds
        
        # Generate expected returns
        np.random.seed(42)
        expected_returns = pd.Series(
            np.random.normal(0.08, 0.04, len(funds)),
            index=funds
        )
        
        # Generate covariance matrix
        correlations = np.random.uniform(0.1, 0.7, (len(funds), len(funds)))
        np.fill_diagonal(correlations, 1.0)
        volatilities = np.random.uniform(0.1, 0.3, len(funds))
        
        covariance_matrix = pd.DataFrame(
            np.outer(volatilities, volatilities) * correlations,
            index=funds,
            columns=funds
        )
        
        return expected_returns, covariance_matrix
    
    async def _get_portfolio_data(self, allocation):
        """Get portfolio-specific data."""
        return {"mock": "data"}
    
    async def _calculate_portfolio_metrics(self, allocation, data):
        """Calculate portfolio metrics."""
        return {"mock": "metrics"}
    
    def _get_portfolio_summary(self, allocation):
        """Generate portfolio summary."""
        total_weight = sum(allocation.values())
        num_funds = len(allocation)
        
        return {
            "total_allocation": total_weight,
            "number_of_funds": num_funds,
            "largest_position": max(allocation.values()) if allocation else 0,
            "fund_names": [TRADEABLE_FUNDS.get(f, {}).get("name", f) for f in allocation.keys()]
        }
    
    def _generate_default_stress_scenarios(self):
        """Generate default Swedish stress scenarios."""
        return [
            {
                "name": "Financial Crisis",
                "description": "2008-style financial crisis",
                "return_shocks": {fund: -0.30 for fund in TRADEABLE_FUNDS.keys()},
                "volatility_multipliers": {fund: 2.0 for fund in TRADEABLE_FUNDS.keys()}
            },
            {
                "name": "Swedish Housing Crash",
                "description": "Swedish housing market collapse",
                "return_shocks": {
                    "PLUS_ALLABOLAG_SVERIGE_INDEX": -0.40,
                    "PLUS_FASTIGHETER_SVERIGE_INDEX": -0.50
                },
                "volatility_multipliers": {
                    "PLUS_ALLABOLAG_SVERIGE_INDEX": 2.5,
                    "PLUS_FASTIGHETER_SVERIGE_INDEX": 3.0
                }
            }
        ]
    
    # Placeholder implementations for other handlers
    async def _handle_portfolio_backtest(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle portfolio backtesting (placeholder)."""
        result = {"message": "Backtesting not fully implemented in this demo"}
        response = self.formatter.success_response(result)
        return [TextContent(type="text", text=json.dumps(response, indent=2))]
    
    async def _handle_efficient_frontier_analysis(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle efficient frontier analysis (placeholder)."""
        result = {"message": "Efficient frontier analysis not fully implemented in this demo"}
        response = self.formatter.success_response(result)
        return [TextContent(type="text", text=json.dumps(response, indent=2))]
    
    async def _handle_portfolio_rebalancing_advice(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle portfolio rebalancing advice (placeholder)."""
        result = {"message": "Rebalancing advice not fully implemented in this demo"}
        response = self.formatter.success_response(result)
        return [TextContent(type="text", text=json.dumps(response, indent=2))]
    
    async def run(self):
        """Run the enhanced MCP server."""
        
        self.logger.info("Starting Enhanced Investment MCP Server")
        
        # Initialize server
        if not await self.initialize_server():
            self.logger.error("Failed to initialize server")
            return
        
        self.logger.info("Enhanced Investment MCP Server is ready")
        
        # Keep server running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Shutting down Enhanced Investment MCP Server")


# Factory function
def create_enhanced_investment_mcp_server() -> EnhancedInvestmentMCPServer:
    """Create and configure an Enhanced Investment MCP server instance."""
    return EnhancedInvestmentMCPServer()