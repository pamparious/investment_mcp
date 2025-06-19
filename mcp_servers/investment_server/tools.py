"""Investment MCP Server Tools."""

import yfinance as yf
import pandas as pd
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MarketDataTool:
    """Tool for fetching market data."""
    
    async def fetch_stock_data(self, symbol: str, period: str = "1mo") -> Dict[str, Any]:
        """Fetch stock data using yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                return {"error": f"No data found for symbol: {symbol}"}
            
            # Convert to dict for JSON serialization
            result = {
                "symbol": symbol,
                "period": period,
                "data": {
                    "latest_close": float(data['Close'].iloc[-1]),
                    "latest_volume": int(data['Volume'].iloc[-1]),
                    "high_52w": float(data['High'].max()),
                    "low_52w": float(data['Low'].min()),
                    "data_points": len(data)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return {"error": str(e)}

# Global instance
market_data_tool = MarketDataTool()