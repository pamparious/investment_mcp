import yfinance as yf
import pandas as pd
from typing import List, Dict, Any, Optional


class StockMarketCollector:
    def __init__(self):
        self.session = None
    
    def get_market_data(
        self, 
        symbols: List[str], 
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict[str, Any]:
        """
        Fetch market data for given symbols.
        
        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'GOOGL'])
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            Dictionary containing market data for each symbol
        """
        result = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval=interval)
                
                if not hist.empty:
                    result[symbol] = {
                        'data': hist.to_dict('index'),
                        'info': self._get_ticker_info(ticker),
                        'symbol': symbol
                    }
                else:
                    result[symbol] = {
                        'data': {},
                        'info': {},
                        'symbol': symbol,
                        'error': 'No data available'
                    }
                    
            except Exception as e:
                result[symbol] = {
                    'data': {},
                    'info': {},
                    'symbol': symbol,
                    'error': str(e)
                }
        
        return result
    
    def _get_ticker_info(self, ticker) -> Dict[str, Any]:
        """
        Get basic ticker information safely.
        """
        try:
            info = ticker.info
            return {
                'longName': info.get('longName', ''),
                'symbol': info.get('symbol', ''),
                'currency': info.get('currency', ''),
                'exchange': info.get('exchange', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'marketCap': info.get('marketCap'),
                'previousClose': info.get('previousClose'),
                'regularMarketPrice': info.get('regularMarketPrice')
            }
        except Exception:
            return {}
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a single symbol.
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception:
            pass
        return None
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """
        Get current prices for multiple symbols.
        """
        return {symbol: self.get_current_price(symbol) for symbol in symbols}