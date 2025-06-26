"""
Unified data collection package for Investment MCP System.

This package consolidates all data collection functionality from various 
scattered collector modules into a streamlined, efficient system.
"""

from .base import BaseDataCollector
from .market_data import MarketDataCollector
from .swedish_data import SwedishDataCollector

__all__ = [
    'BaseDataCollector',
    'MarketDataCollector', 
    'SwedishDataCollector'
]