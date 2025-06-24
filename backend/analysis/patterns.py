"""Technical analysis patterns and indicators."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Technical analysis calculations and pattern detection."""
    
    def __init__(self):
        """Initialize the technical analyzer."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def prepare_dataframe(self, market_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Prepare pandas DataFrame from market data.
        
        Args:
            market_data: List of market data dictionaries
            
        Returns:
            Prepared pandas DataFrame
        """
        if not market_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(market_data)
        
        # Ensure required columns exist
        required_cols = ['date', 'close_price', 'high_price', 'low_price', 'open_price', 'volume']
        for col in required_cols:
            if col not in df.columns:
                self.logger.warning(f"Missing required column: {col}")
                return pd.DataFrame()
        
        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df.set_index('date', inplace=True)
        
        # Ensure numeric columns
        numeric_cols = ['close_price', 'high_price', 'low_price', 'open_price', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.dropna()
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Simple and Exponential Moving Averages.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing moving averages
        """
        if df.empty or 'close_price' not in df.columns:
            return {}
        
        try:
            result = {}
            
            # Simple Moving Averages
            if len(df) >= 20:
                result['sma_20'] = df['close_price'].rolling(window=20).mean().iloc[-1]
            if len(df) >= 50:
                result['sma_50'] = df['close_price'].rolling(window=50).mean().iloc[-1]
            if len(df) >= 200:
                result['sma_200'] = df['close_price'].rolling(window=200).mean().iloc[-1]
            
            # Exponential Moving Averages
            if len(df) >= 12:
                result['ema_12'] = df['close_price'].ewm(span=12).mean().iloc[-1]
            if len(df) >= 26:
                result['ema_26'] = df['close_price'].ewm(span=26).mean().iloc[-1]
            
            # Current price for comparison
            result['current_price'] = df['close_price'].iloc[-1]
            
            # Moving average signals
            if 'sma_20' in result and 'sma_50' in result:
                result['golden_cross'] = result['sma_20'] > result['sma_50']
                result['death_cross'] = result['sma_20'] < result['sma_50']
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating moving averages: {e}")
            return {}
    
    def detect_trends(self, df: pd.DataFrame, period: int = 20) -> Dict[str, Any]:
        """
        Detect trend direction and strength.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for trend analysis
            
        Returns:
            Dictionary containing trend analysis
        """
        if df.empty or len(df) < period:
            return {"trend": "insufficient_data", "strength": 0}
        
        try:
            # Calculate trend using linear regression
            close_prices = df['close_price'].tail(period)
            x = np.arange(len(close_prices))
            slope, intercept = np.polyfit(x, close_prices, 1)
            
            # Calculate R-squared for trend strength
            correlation = np.corrcoef(x, close_prices)[0, 1]
            r_squared = correlation ** 2
            
            # Determine trend direction
            if slope > 0:
                trend = "bullish"
            elif slope < 0:
                trend = "bearish"
            else:
                trend = "sideways"
            
            # Calculate percentage change
            start_price = close_prices.iloc[0]
            end_price = close_prices.iloc[-1]
            pct_change = ((end_price - start_price) / start_price) * 100
            
            return {
                "trend": trend,
                "strength": float(r_squared),
                "slope": float(slope),
                "percentage_change": float(pct_change),
                "period_days": period,
                "confidence": "high" if r_squared > 0.7 else "medium" if r_squared > 0.4 else "low"
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting trends: {e}")
            return {"trend": "error", "strength": 0}
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """
        Calculate Relative Strength Index.
        
        Args:
            df: DataFrame with OHLCV data
            period: RSI calculation period
            
        Returns:
            RSI value or None if insufficient data
        """
        if df.empty or len(df) < period + 1:
            return None
        
        try:
            delta = df['close_price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1])
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return None
    
    def find_support_resistance(self, df: pd.DataFrame, window: int = 10) -> Dict[str, Any]:
        """
        Find support and resistance levels.
        
        Args:
            df: DataFrame with OHLCV data
            window: Window for local minima/maxima detection
            
        Returns:
            Dictionary containing support and resistance levels
        """
        if df.empty or len(df) < window * 2:
            return {}
        
        try:
            # Find local minima (support) and maxima (resistance)
            highs = df['high_price']
            lows = df['low_price']
            
            # Simple support/resistance using rolling min/max
            recent_period = min(50, len(df))
            recent_data = df.tail(recent_period)
            
            support_level = recent_data['low_price'].min()
            resistance_level = recent_data['high_price'].max()
            
            current_price = df['close_price'].iloc[-1]
            
            # Calculate distances from current price
            support_distance = ((current_price - support_level) / current_price) * 100
            resistance_distance = ((resistance_level - current_price) / current_price) * 100
            
            return {
                "support": float(support_level),
                "resistance": float(resistance_level),
                "current_price": float(current_price),
                "support_distance_pct": float(support_distance),
                "resistance_distance_pct": float(resistance_distance),
                "near_support": support_distance < 5,  # Within 5% of support
                "near_resistance": resistance_distance < 5  # Within 5% of resistance
            }
            
        except Exception as e:
            self.logger.error(f"Error finding support/resistance: {e}")
            return {}
    
    def calculate_volatility(self, df: pd.DataFrame, period: int = 20) -> Dict[str, Any]:
        """
        Calculate price volatility metrics.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for volatility calculation
            
        Returns:
            Dictionary containing volatility metrics
        """
        if df.empty or len(df) < period:
            return {}
        
        try:
            # Calculate daily returns
            returns = df['close_price'].pct_change().dropna()
            
            if len(returns) < period:
                return {}
            
            # Historical volatility (annualized)
            volatility = returns.std() * np.sqrt(252)  # Assuming 252 trading days
            
            # Average True Range (ATR)
            high_low = df['high_price'] - df['low_price']
            high_close = np.abs(df['high_price'] - df['close_price'].shift())
            low_close = np.abs(df['low_price'] - df['close_price'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            # Volatility classification
            if volatility < 0.2:
                vol_class = "low"
            elif volatility < 0.4:
                vol_class = "medium"
            else:
                vol_class = "high"
            
            return {
                "historical_volatility": float(volatility),
                "atr": float(atr),
                "volatility_class": vol_class,
                "period_days": period
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return {}
    
    def analyze_symbol(self, symbol: str, market_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis for a symbol.
        
        Args:
            symbol: Stock symbol
            market_data: List of market data dictionaries
            
        Returns:
            Complete technical analysis results
        """
        try:
            df = self.prepare_dataframe(market_data)
            
            if df.empty:
                return {
                    "symbol": symbol,
                    "error": "Insufficient or invalid data for analysis",
                    "data_points": 0
                }
            
            # Perform all analyses
            moving_averages = self.calculate_moving_averages(df)
            trends = self.detect_trends(df)
            rsi = self.calculate_rsi(df)
            support_resistance = self.find_support_resistance(df)
            volatility = self.calculate_volatility(df)
            
            # Combine results
            analysis = {
                "symbol": symbol,
                "data_points": len(df),
                "analysis_date": datetime.now().isoformat(),
                "moving_averages": moving_averages,
                "trend_analysis": trends,
                "rsi": rsi,
                "support_resistance": support_resistance,
                "volatility": volatility
            }
            
            # Add overall technical score
            analysis["technical_score"] = self._calculate_technical_score(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing symbol {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "data_points": 0
            }
    
    def _calculate_technical_score(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall technical score based on indicators.
        
        Args:
            analysis: Technical analysis results
            
        Returns:
            Technical score summary
        """
        try:
            score = 0
            max_score = 0
            signals = []
            
            # RSI signals
            rsi = analysis.get("rsi")
            if rsi:
                max_score += 2
                if rsi < 30:
                    score += 2
                    signals.append("RSI oversold (bullish)")
                elif rsi > 70:
                    score -= 2
                    signals.append("RSI overbought (bearish)")
                elif 40 <= rsi <= 60:
                    score += 1
                    signals.append("RSI neutral")
            
            # Trend signals
            trend = analysis.get("trend_analysis", {})
            if trend.get("trend"):
                max_score += 3
                if trend["trend"] == "bullish" and trend.get("strength", 0) > 0.5:
                    score += 3
                    signals.append("Strong bullish trend")
                elif trend["trend"] == "bullish":
                    score += 1
                    signals.append("Weak bullish trend")
                elif trend["trend"] == "bearish" and trend.get("strength", 0) > 0.5:
                    score -= 3
                    signals.append("Strong bearish trend")
                elif trend["trend"] == "bearish":
                    score -= 1
                    signals.append("Weak bearish trend")
            
            # Moving average signals
            ma = analysis.get("moving_averages", {})
            if "golden_cross" in ma or "death_cross" in ma:
                max_score += 2
                if ma.get("golden_cross"):
                    score += 2
                    signals.append("Golden cross (bullish)")
                elif ma.get("death_cross"):
                    score -= 2
                    signals.append("Death cross (bearish)")
            
            # Support/Resistance signals
            sr = analysis.get("support_resistance", {})
            if sr:
                max_score += 1
                if sr.get("near_support"):
                    score += 1
                    signals.append("Near support level (potential bounce)")
                elif sr.get("near_resistance"):
                    score -= 1
                    signals.append("Near resistance level (potential reversal)")
            
            # Normalize score to -100 to +100 scale
            if max_score > 0:
                normalized_score = (score / max_score) * 100
            else:
                normalized_score = 0
            
            # Determine overall signal
            if normalized_score > 50:
                overall_signal = "bullish"
            elif normalized_score < -50:
                overall_signal = "bearish"
            else:
                overall_signal = "neutral"
            
            return {
                "score": round(normalized_score, 1),
                "signal": overall_signal,
                "signals": signals,
                "max_possible_score": max_score
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating technical score: {e}")
            return {
                "score": 0,
                "signal": "neutral",
                "signals": ["Error calculating score"],
                "error": str(e)
            }