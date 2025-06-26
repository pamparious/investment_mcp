"""
Unified technical analysis module for Investment MCP System.

This module consolidates all technical analysis functionality into a single,
comprehensive system for calculating technical indicators and patterns.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Comprehensive technical analysis for market data."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    def calculate_moving_averages(
        self, 
        prices: pd.Series, 
        windows: List[int] = [20, 50, 200]
    ) -> Dict[str, pd.Series]:
        """
        Calculate simple and exponential moving averages.
        
        Args:
            prices: Series of closing prices
            windows: List of window sizes for moving averages
            
        Returns:
            Dictionary containing moving average series
        """
        
        if prices.empty or len(prices) < max(windows):
            return {}
        
        try:
            moving_averages = {}
            
            for window in windows:
                if len(prices) >= window:
                    # Simple Moving Average
                    sma = prices.rolling(window=window).mean()
                    moving_averages[f"SMA_{window}"] = sma
                    
                    # Exponential Moving Average
                    ema = prices.ewm(span=window).mean()
                    moving_averages[f"EMA_{window}"] = ema
            
            return moving_averages
            
        except Exception as e:
            self.logger.error(f"Error calculating moving averages: {e}")
            return {}
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Series of closing prices
            window: Period for RSI calculation
            
        Returns:
            Series of RSI values
        """
        
        if prices.empty or len(prices) < window + 1:
            return pd.Series()
        
        try:
            # Calculate price changes
            delta = prices.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gains = gains.rolling(window=window).mean()
            avg_losses = losses.rolling(window=window).mean()
            
            # Calculate RS and RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)  # Fill NaN with neutral RSI
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return pd.Series()
    
    def calculate_macd(
        self, 
        prices: pd.Series, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Series of closing prices
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            Dictionary containing MACD line, signal line, and histogram
        """
        
        if prices.empty or len(prices) < slow_period + signal_period:
            return {}
        
        try:
            # Calculate EMAs
            ema_fast = prices.ewm(span=fast_period).mean()
            ema_slow = prices.ewm(span=slow_period).mean()
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=signal_period).mean()
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return {
                "MACD": macd_line,
                "Signal": signal_line,
                "Histogram": histogram
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return {}
    
    def calculate_bollinger_bands(
        self, 
        prices: pd.Series, 
        window: int = 20, 
        num_std: float = 2.0
    ) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Series of closing prices
            window: Period for moving average
            num_std: Number of standard deviations for bands
            
        Returns:
            Dictionary containing upper band, middle band (SMA), and lower band
        """
        
        if prices.empty or len(prices) < window:
            return {}
        
        try:
            # Calculate middle band (SMA)
            middle_band = prices.rolling(window=window).mean()
            
            # Calculate standard deviation
            std_dev = prices.rolling(window=window).std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (std_dev * num_std)
            lower_band = middle_band - (std_dev * num_std)
            
            # Calculate %B (position within bands)
            percent_b = (prices - lower_band) / (upper_band - lower_band)
            
            # Calculate bandwidth
            bandwidth = (upper_band - lower_band) / middle_band
            
            return {
                "Upper_Band": upper_band,
                "Middle_Band": middle_band,
                "Lower_Band": lower_band,
                "Percent_B": percent_b,
                "Bandwidth": bandwidth
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            return {}
    
    def calculate_volatility_indicators(self, prices: pd.Series, window: int = 20) -> Dict[str, Any]:
        """
        Calculate various volatility indicators.
        
        Args:
            prices: Series of closing prices
            window: Period for calculations
            
        Returns:
            Dictionary containing volatility metrics
        """
        
        if prices.empty or len(prices) < window:
            return {}
        
        try:
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Historical volatility (annualized)
            historical_vol = returns.rolling(window=window).std() * np.sqrt(252)
            
            # Average True Range (simplified version using close prices)
            high_low = prices.rolling(window=2).max() - prices.rolling(window=2).min()
            atr = high_low.rolling(window=window).mean()
            
            # Volatility ratio (current vs average)
            current_vol = returns.rolling(window=window).std().iloc[-1] if len(returns) >= window else 0
            avg_vol = returns.std()
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
            
            return {
                "historical_volatility": historical_vol,
                "atr": atr,
                "current_volatility": float(current_vol),
                "average_volatility": float(avg_vol),
                "volatility_ratio": float(vol_ratio),
                "volatility_percentile": float(stats.percentileofscore(historical_vol.dropna(), current_vol)) if len(historical_vol.dropna()) > 0 else 50
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility indicators: {e}")
            return {}
    
    def detect_trend_direction(self, prices: pd.Series, window: int = 20) -> Dict[str, Any]:
        """
        Detect trend direction using multiple methods.
        
        Args:
            prices: Series of closing prices
            window: Period for trend analysis
            
        Returns:
            Dictionary containing trend analysis
        """
        
        if prices.empty or len(prices) < window:
            return {}
        
        try:
            # Method 1: Linear regression trend
            recent_prices = prices.tail(window)
            x = np.arange(len(recent_prices))
            slope, intercept = np.polyfit(x, recent_prices, 1)
            trend_strength = abs(slope) / recent_prices.mean() * 100  # Normalize by price level
            
            # Method 2: Moving average trend
            sma_short = prices.rolling(window=window//2).mean()
            sma_long = prices.rolling(window=window).mean()
            ma_trend = "bullish" if sma_short.iloc[-1] > sma_long.iloc[-1] else "bearish"
            
            # Method 3: Price momentum
            momentum = (prices.iloc[-1] - prices.iloc[-window]) / prices.iloc[-window] * 100
            
            # Method 4: Higher highs, higher lows analysis
            recent_highs = prices.rolling(window=5).max().tail(window//2)
            recent_lows = prices.rolling(window=5).min().tail(window//2)
            
            higher_highs = (recent_highs.diff() > 0).sum()
            higher_lows = (recent_lows.diff() > 0).sum()
            
            hh_hl_score = (higher_highs + higher_lows) / (len(recent_highs) - 1) if len(recent_highs) > 1 else 0.5
            
            # Combine signals
            bullish_signals = 0
            bearish_signals = 0
            
            if slope > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            if ma_trend == "bullish":
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            if momentum > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            if hh_hl_score > 0.5:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # Determine overall trend
            if bullish_signals > bearish_signals:
                overall_trend = "bullish"
                trend_confidence = bullish_signals / (bullish_signals + bearish_signals)
            elif bearish_signals > bullish_signals:
                overall_trend = "bearish"
                trend_confidence = bearish_signals / (bullish_signals + bearish_signals)
            else:
                overall_trend = "neutral"
                trend_confidence = 0.5
            
            return {
                "overall_trend": overall_trend,
                "trend_confidence": float(trend_confidence),
                "trend_strength": float(trend_strength),
                "slope": float(slope),
                "momentum_pct": float(momentum),
                "ma_trend": ma_trend,
                "hh_hl_score": float(hh_hl_score),
                "bullish_signals": int(bullish_signals),
                "bearish_signals": int(bearish_signals)
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting trend direction: {e}")
            return {}
    
    def identify_support_resistance(self, prices: pd.Series, window: int = 20) -> Dict[str, Any]:
        """
        Identify potential support and resistance levels.
        
        Args:
            prices: Series of closing prices
            window: Period for identifying levels
            
        Returns:
            Dictionary containing support and resistance levels
        """
        
        if prices.empty or len(prices) < window * 2:
            return {}
        
        try:
            # Find local peaks and troughs
            peaks = []
            troughs = []
            
            for i in range(window, len(prices) - window):
                # Check if current price is a local maximum
                if prices.iloc[i] == prices.iloc[i-window:i+window+1].max():
                    peaks.append(prices.iloc[i])
                
                # Check if current price is a local minimum
                if prices.iloc[i] == prices.iloc[i-window:i+window+1].min():
                    troughs.append(prices.iloc[i])
            
            # Calculate support and resistance levels
            resistance_levels = []
            support_levels = []
            
            if peaks:
                # Group similar peaks (within 2% of each other)
                peaks_sorted = sorted(peaks, reverse=True)
                current_level = peaks_sorted[0]
                level_count = 1
                
                for peak in peaks_sorted[1:]:
                    if abs(peak - current_level) / current_level <= 0.02:  # Within 2%
                        level_count += 1
                    else:
                        if level_count >= 2:  # At least 2 touches
                            resistance_levels.append({
                                "level": float(current_level),
                                "touches": level_count,
                                "strength": min(level_count / len(peaks) * 100, 100)
                            })
                        current_level = peak
                        level_count = 1
                
                # Add the last level
                if level_count >= 2:
                    resistance_levels.append({
                        "level": float(current_level),
                        "touches": level_count,
                        "strength": min(level_count / len(peaks) * 100, 100)
                    })
            
            if troughs:
                # Group similar troughs
                troughs_sorted = sorted(troughs)
                current_level = troughs_sorted[0]
                level_count = 1
                
                for trough in troughs_sorted[1:]:
                    if abs(trough - current_level) / current_level <= 0.02:  # Within 2%
                        level_count += 1
                    else:
                        if level_count >= 2:  # At least 2 touches
                            support_levels.append({
                                "level": float(current_level),
                                "touches": level_count,
                                "strength": min(level_count / len(troughs) * 100, 100)
                            })
                        current_level = trough
                        level_count = 1
                
                # Add the last level
                if level_count >= 2:
                    support_levels.append({
                        "level": float(current_level),
                        "touches": level_count,
                        "strength": min(level_count / len(troughs) * 100, 100)
                    })
            
            # Get current price for distance calculations
            current_price = float(prices.iloc[-1])
            
            # Calculate distances to nearest levels
            nearest_resistance = None
            nearest_support = None
            
            if resistance_levels:
                resistance_above = [r for r in resistance_levels if r["level"] > current_price]
                if resistance_above:
                    nearest_resistance = min(resistance_above, key=lambda x: x["level"])
            
            if support_levels:
                support_below = [s for s in support_levels if s["level"] < current_price]
                if support_below:
                    nearest_support = max(support_below, key=lambda x: x["level"])
            
            return {
                "resistance_levels": resistance_levels,
                "support_levels": support_levels,
                "nearest_resistance": nearest_resistance,
                "nearest_support": nearest_support,
                "current_price": current_price,
                "total_peaks": len(peaks),
                "total_troughs": len(troughs)
            }
            
        except Exception as e:
            self.logger.error(f"Error identifying support/resistance: {e}")
            return {}
    
    def generate_technical_summary(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive technical analysis summary.
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing complete technical analysis
        """
        
        if market_data.empty or 'close_price' not in market_data.columns:
            return {}
        
        try:
            prices = market_data['close_price']
            
            # Calculate all indicators
            moving_averages = self.calculate_moving_averages(prices)
            rsi = self.calculate_rsi(prices)
            macd = self.calculate_macd(prices)
            bollinger = self.calculate_bollinger_bands(prices)
            volatility = self.calculate_volatility_indicators(prices)
            trend = self.detect_trend_direction(prices)
            support_resistance = self.identify_support_resistance(prices)
            
            # Current values
            current_price = float(prices.iloc[-1])
            current_rsi = float(rsi.iloc[-1]) if not rsi.empty else 50
            
            # Generate signals
            signals = []
            
            # RSI signals
            if current_rsi > 70:
                signals.append({"type": "sell", "indicator": "RSI", "strength": "medium", "reason": "Overbought condition"})
            elif current_rsi < 30:
                signals.append({"type": "buy", "indicator": "RSI", "strength": "medium", "reason": "Oversold condition"})
            
            # Moving average signals
            if moving_averages:
                sma_20 = moving_averages.get("SMA_20")
                sma_50 = moving_averages.get("SMA_50")
                
                if sma_20 is not None and sma_50 is not None and not sma_20.empty and not sma_50.empty:
                    if sma_20.iloc[-1] > sma_50.iloc[-1] and sma_20.iloc[-2] <= sma_50.iloc[-2]:
                        signals.append({"type": "buy", "indicator": "MA_Cross", "strength": "strong", "reason": "Golden cross"})
                    elif sma_20.iloc[-1] < sma_50.iloc[-1] and sma_20.iloc[-2] >= sma_50.iloc[-2]:
                        signals.append({"type": "sell", "indicator": "MA_Cross", "strength": "strong", "reason": "Death cross"})
            
            # MACD signals
            if macd and "MACD" in macd and "Signal" in macd:
                macd_line = macd["MACD"]
                signal_line = macd["Signal"]
                
                if not macd_line.empty and not signal_line.empty and len(macd_line) > 1:
                    if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
                        signals.append({"type": "buy", "indicator": "MACD", "strength": "medium", "reason": "MACD bullish crossover"})
                    elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
                        signals.append({"type": "sell", "indicator": "MACD", "strength": "medium", "reason": "MACD bearish crossover"})
            
            # Bollinger Bands signals
            if bollinger and "Percent_B" in bollinger:
                percent_b = bollinger["Percent_B"]
                if not percent_b.empty:
                    current_percent_b = percent_b.iloc[-1]
                    if current_percent_b > 1:
                        signals.append({"type": "sell", "indicator": "Bollinger", "strength": "medium", "reason": "Price above upper band"})
                    elif current_percent_b < 0:
                        signals.append({"type": "buy", "indicator": "Bollinger", "strength": "medium", "reason": "Price below lower band"})
            
            # Calculate overall signal strength
            buy_signals = [s for s in signals if s["type"] == "buy"]
            sell_signals = [s for s in signals if s["type"] == "sell"]
            
            signal_strength_map = {"weak": 1, "medium": 2, "strong": 3}
            buy_strength = sum(signal_strength_map.get(s["strength"], 1) for s in buy_signals)
            sell_strength = sum(signal_strength_map.get(s["strength"], 1) for s in sell_signals)
            
            if buy_strength > sell_strength:
                overall_signal = "buy"
                signal_confidence = buy_strength / (buy_strength + sell_strength)
            elif sell_strength > buy_strength:
                overall_signal = "sell"
                signal_confidence = sell_strength / (buy_strength + sell_strength)
            else:
                overall_signal = "hold"
                signal_confidence = 0.5
            
            return {
                "current_price": current_price,
                "current_rsi": current_rsi,
                "moving_averages": {k: float(v.iloc[-1]) if not v.empty else None for k, v in moving_averages.items()},
                "macd_current": {k: float(v.iloc[-1]) if not v.empty else None for k, v in macd.items()} if macd else {},
                "bollinger_current": {k: float(v.iloc[-1]) if not v.empty else None for k, v in bollinger.items()} if bollinger else {},
                "volatility": volatility,
                "trend_analysis": trend,
                "support_resistance": support_resistance,
                "signals": signals,
                "overall_signal": overall_signal,
                "signal_confidence": float(signal_confidence),
                "buy_signals_count": len(buy_signals),
                "sell_signals_count": len(sell_signals)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating technical summary: {e}")
            return {}