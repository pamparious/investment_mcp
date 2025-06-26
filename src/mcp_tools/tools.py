"""
MCP tools and utilities for Investment MCP System.

This module provides utility functions and helper classes for MCP tool
implementations and server management.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from ..core.config import TRADEABLE_FUNDS, validate_fund_allocation

logger = logging.getLogger(__name__)


class MCPToolValidator:
    """Validator for MCP tool inputs and outputs."""
    
    @staticmethod
    def validate_fund_ids(fund_ids: List[str]) -> Dict[str, Any]:
        """Validate that fund IDs are in the approved universe."""
        
        valid_funds = []
        invalid_funds = []
        
        for fund_id in fund_ids:
            if fund_id in TRADEABLE_FUNDS:
                valid_funds.append(fund_id)
            else:
                invalid_funds.append(fund_id)
        
        return {
            "valid": len(invalid_funds) == 0,
            "valid_funds": valid_funds,
            "invalid_funds": invalid_funds,
            "total_requested": len(fund_ids),
            "total_valid": len(valid_funds)
        }
    
    @staticmethod
    def validate_portfolio_allocation(allocation: Dict[str, float]) -> Dict[str, Any]:
        """Validate portfolio allocation using core validation."""
        return validate_fund_allocation(allocation)
    
    @staticmethod
    def validate_risk_tolerance(risk_tolerance: str) -> bool:
        """Validate risk tolerance parameter."""
        return risk_tolerance in ["low", "medium", "high", "very_high"]
    
    @staticmethod
    def validate_investment_horizon(horizon: str) -> bool:
        """Validate investment horizon parameter."""
        return horizon in ["short", "medium", "long"]
    
    @staticmethod
    def validate_investment_amount(amount: float) -> Dict[str, Any]:
        """Validate investment amount."""
        
        min_amount = 1000  # Minimum 1000 SEK
        max_amount = 100000000  # Maximum 100M SEK
        
        return {
            "valid": min_amount <= amount <= max_amount,
            "amount": amount,
            "min_amount": min_amount,
            "max_amount": max_amount,
            "currency": "SEK"
        }


class MCPResponseFormatter:
    """Formatter for standardized MCP tool responses."""
    
    @staticmethod
    def success_response(data: Any, message: Optional[str] = None) -> Dict[str, Any]:
        """Create standardized success response."""
        
        response = {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        if message:
            response["message"] = message
        
        return response
    
    @staticmethod
    def error_response(error: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create standardized error response."""
        
        response = {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": error
        }
        
        if details:
            response["error_details"] = details
        
        return response
    
    @staticmethod
    def validation_error_response(validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create validation error response."""
        
        return {
            "status": "validation_error",
            "timestamp": datetime.utcnow().isoformat(),
            "validation_errors": validation_result.get("errors", []),
            "validation_details": validation_result
        }
    
    @staticmethod
    def format_fund_data(fund_data: Dict[str, Any], include_metadata: bool = True) -> Dict[str, Any]:
        """Format fund data for MCP response."""
        
        formatted = {}
        
        for fund_id, fund_info in fund_data.items():
            formatted[fund_id] = {
                "name": fund_info.get("name", fund_id),
                "category": fund_info.get("category", "unknown"),
                "risk_level": fund_info.get("risk_level", "medium"),
                "expense_ratio": fund_info.get("expense_ratio", 0.0),
                "currency": fund_info.get("currency", "SEK")
            }
            
            if include_metadata:
                formatted[fund_id]["description"] = fund_info.get("description", "")
                formatted[fund_id]["type"] = fund_info.get("type", "")
        
        return formatted
    
    @staticmethod
    def format_portfolio_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Format portfolio metrics for display."""
        
        formatted = {}
        
        # Format percentages
        percentage_fields = [
            "expected_return", "expected_volatility", "annualized_return", 
            "annualized_volatility", "max_drawdown"
        ]
        
        for field in percentage_fields:
            if field in metrics:
                formatted[f"{field}_percent"] = f"{metrics[field] * 100:.2f}%"
        
        # Format ratios
        ratio_fields = ["sharpe_ratio", "sortino_ratio"]
        for field in ratio_fields:
            if field in metrics:
                formatted[field] = f"{metrics[field]:.3f}"
        
        # Keep original values for programmatic use
        formatted["raw_metrics"] = metrics
        
        return formatted


class MCPDataProcessor:
    """Processor for converting data between formats for MCP tools."""
    
    @staticmethod
    def portfolio_to_allocation_list(portfolio: Dict[str, float]) -> List[Dict[str, Any]]:
        """Convert portfolio dict to allocation list format."""
        
        allocation_list = []
        
        for fund_id, weight in portfolio.items():
            fund_info = TRADEABLE_FUNDS.get(fund_id, {})
            
            allocation_list.append({
                "fund_id": fund_id,
                "fund_name": fund_info.get("name", fund_id),
                "allocation_percent": weight * 100,
                "allocation_decimal": weight,
                "category": fund_info.get("category", "unknown"),
                "risk_level": fund_info.get("risk_level", "medium")
            })
        
        # Sort by allocation descending
        allocation_list.sort(key=lambda x: x["allocation_decimal"], reverse=True)
        
        return allocation_list
    
    @staticmethod
    def allocation_list_to_portfolio(allocation_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """Convert allocation list to portfolio dict format."""
        
        portfolio = {}
        
        for allocation in allocation_list:
            fund_id = allocation.get("fund_id")
            weight = allocation.get("allocation_decimal") or allocation.get("allocation_percent", 0) / 100
            
            if fund_id and weight > 0:
                portfolio[fund_id] = weight
        
        return portfolio
    
    @staticmethod
    def calculate_portfolio_summary(portfolio: Dict[str, float]) -> Dict[str, Any]:
        """Calculate portfolio summary statistics."""
        
        if not portfolio:
            return {"error": "Empty portfolio"}
        
        total_weight = sum(portfolio.values())
        num_funds = len(portfolio)
        max_allocation = max(portfolio.values()) if portfolio else 0
        min_allocation = min(portfolio.values()) if portfolio else 0
        
        # Categorize by fund types
        categories = {}
        risk_levels = {}
        
        for fund_id, weight in portfolio.items():
            fund_info = TRADEABLE_FUNDS.get(fund_id, {})
            category = fund_info.get("category", "unknown")
            risk_level = fund_info.get("risk_level", "medium")
            
            categories[category] = categories.get(category, 0) + weight
            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + weight
        
        return {
            "total_allocation": total_weight,
            "number_of_funds": num_funds,
            "concentration": {
                "max_allocation": max_allocation,
                "min_allocation": min_allocation,
                "concentration_ratio": max_allocation,
                "diversification_score": 1 - (sum(w**2 for w in portfolio.values()))
            },
            "category_breakdown": categories,
            "risk_level_breakdown": risk_levels,
            "fund_count_by_category": len(set(TRADEABLE_FUNDS.get(f, {}).get("category") for f in portfolio.keys())),
            "is_well_diversified": num_funds >= 3 and max_allocation <= 0.5
        }


class MCPToolRegistry:
    """Registry for managing available MCP tools and their metadata."""
    
    def __init__(self):
        self.tools = {}
        self.tool_categories = {
            "data_collection": ["collect_fund_data", "collect_swedish_economic_data"],
            "analysis": ["analyze_fund_risk", "technical_analysis", "market_sentiment_analysis"],
            "optimization": ["optimize_portfolio", "stress_test_portfolio"],
            "ai_services": ["ai_portfolio_recommendation", "explain_fund_selection"],
            "utilities": ["get_fund_universe"]
        }
    
    def get_tools_by_category(self, category: str) -> List[str]:
        """Get tools by category."""
        return self.tool_categories.get(category, [])
    
    def get_all_categories(self) -> List[str]:
        """Get all available tool categories."""
        return list(self.tool_categories.keys())
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific tool."""
        
        tool_descriptions = {
            "collect_fund_data": {
                "description": "Collect historical market data for approved Swedish funds",
                "category": "data_collection",
                "input_types": ["fund_ids", "years"],
                "output_type": "market_data_summary",
                "estimated_duration": "30-120 seconds"
            },
            "collect_swedish_economic_data": {
                "description": "Collect Swedish economic indicators from Riksbanken and SCB",
                "category": "data_collection", 
                "input_types": ["years"],
                "output_type": "economic_data_summary",
                "estimated_duration": "15-60 seconds"
            },
            "analyze_fund_risk": {
                "description": "Perform comprehensive risk analysis for fund or portfolio",
                "category": "analysis",
                "input_types": ["fund_id", "portfolio"],
                "output_type": "risk_metrics",
                "estimated_duration": "5-15 seconds"
            },
            "technical_analysis": {
                "description": "Perform technical analysis on fund price data",
                "category": "analysis",
                "input_types": ["fund_id", "indicators"],
                "output_type": "technical_indicators",
                "estimated_duration": "5-10 seconds"
            },
            "optimize_portfolio": {
                "description": "Optimize portfolio allocation using Modern Portfolio Theory",
                "category": "optimization",
                "input_types": ["risk_tolerance", "optimization_method", "target_return"],
                "output_type": "optimized_allocation",
                "estimated_duration": "10-30 seconds"
            },
            "ai_portfolio_recommendation": {
                "description": "Generate AI-enhanced portfolio recommendation",
                "category": "ai_services",
                "input_types": ["risk_tolerance", "investment_horizon", "investment_amount"],
                "output_type": "ai_recommendation",
                "estimated_duration": "15-45 seconds"
            },
            "stress_test_portfolio": {
                "description": "Perform stress testing on portfolio under various scenarios",
                "category": "optimization",
                "input_types": ["portfolio", "scenarios"],
                "output_type": "stress_test_results",
                "estimated_duration": "10-20 seconds"
            },
            "get_fund_universe": {
                "description": "Get information about all approved funds",
                "category": "utilities",
                "input_types": ["category"],
                "output_type": "fund_information",
                "estimated_duration": "1-2 seconds"
            },
            "market_sentiment_analysis": {
                "description": "Analyze current market sentiment using latest data",
                "category": "analysis",
                "input_types": ["include_ai_insights"],
                "output_type": "sentiment_analysis",
                "estimated_duration": "10-30 seconds"
            },
            "explain_fund_selection": {
                "description": "Get AI explanation for fund selection reasoning",
                "category": "ai_services",
                "input_types": ["fund_ids"],
                "output_type": "ai_explanation",
                "estimated_duration": "15-30 seconds"
            }
        }
        
        return tool_descriptions.get(tool_name, {"description": "Unknown tool"})
    
    def validate_tool_availability(self) -> Dict[str, Any]:
        """Validate that all required components are available for tools."""
        
        availability = {
            "data_collection": True,  # Basic collectors always available
            "analysis": True,         # Analysis modules always available
            "optimization": True,     # Optimization always available
            "ai_services": False,     # Depends on AI provider availability
            "utilities": True         # Utilities always available
        }
        
        # In production, would check actual component availability
        # For example, check if AI providers are configured and accessible
        
        return {
            "overall_status": "partial" if not all(availability.values()) else "full",
            "category_availability": availability,
            "unavailable_reason": "AI services require configured AI provider" if not availability["ai_services"] else None
        }