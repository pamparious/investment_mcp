"""Technical analysis engine for financial data."""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

from ...utils.exceptions import AnalysisError

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Technical analysis engine with comprehensive indicators."""
    
    def __init__(self):
        """Initialize the technical analyzer."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def calculate_moving_averages(
        self,
        data: pd.DataFrame,
        price_column: str = "Close",
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Calculate various moving averages.
        
        Args:
            data: DataFrame with price data
            price_column: Column name for prices
            windows: List of window sizes for moving averages
            
        Returns:
            DataFrame with moving averages added
        """
        if windows is None:
            windows = [5, 10, 20, 50, 100, 200]
        
        df = data.copy()
        prices = df[price_column]
        
        for window in windows:
            if len(df) >= window:
                # Simple Moving Average
                df[f"sma_{window}"] = prices.rolling(window=window).mean()
                
                # Exponential Moving Average
                df[f"ema_{window}"] = prices.ewm(span=window, adjust=False).mean()
                
                # Weighted Moving Average
                weights = np.arange(1, window + 1)
                df[f"wma_{window}"] = prices.rolling(window).apply(
                    lambda x: np.dot(x, weights) / weights.sum(), raw=True
                )
        
        # Hull Moving Average (more responsive)
        if len(df) >= 20:
            df["hma_20"] = self._calculate_hull_ma(prices, 20)
        
        return df
    
    def calculate_momentum_indicators(
        self,
        data: pd.DataFrame,
        price_column: str = "Close",
        volume_column: str = "Volume"
    ) -> pd.DataFrame:
        """
        Calculate momentum indicators.
        
        Args:
            data: DataFrame with price data
            price_column: Column name for prices
            volume_column: Column name for volume
            
        Returns:
            DataFrame with momentum indicators added
        """
        df = data.copy()
        prices = df[price_column]
        
        # RSI (Relative Strength Index)
        df["rsi_14"] = self._calculate_rsi(prices, window=14)
        df["rsi_30"] = self._calculate_rsi(prices, window=30)
        
        # MACD (Moving Average Convergence Divergence)
        macd_line, macd_signal, macd_histogram = self._calculate_macd(prices)
        df["macd"] = macd_line
        df["macd_signal"] = macd_signal
        df["macd_histogram"] = macd_histogram
        
        # Stochastic Oscillator
        if all(col in df.columns for col in ["High", "Low"]):
            df["stoch_k"], df["stoch_d"] = self._calculate_stochastic(
                df["High"], df["Low"], prices
            )
        
        # Williams %R
        if all(col in df.columns for col in ["High", "Low"]):
            df["williams_r"] = self._calculate_williams_r(
                df["High"], df["Low"], prices
            )
        
        # Rate of Change (ROC)
        df["roc_10"] = self._calculate_roc(prices, 10)
        df["roc_20"] = self._calculate_roc(prices, 20)
        
        # Money Flow Index (if volume available)
        if volume_column in df.columns and all(col in df.columns for col in ["High", "Low"]):
            df["mfi"] = self._calculate_mfi(
                df["High"], df["Low"], prices, df[volume_column]
            )
        
        # Commodity Channel Index
        if all(col in df.columns for col in ["High", "Low"]):
            df["cci"] = self._calculate_cci(df["High"], df["Low"], prices)
        
        return df
    
    def calculate_volatility_indicators(
        self,
        data: pd.DataFrame,
        price_column: str = "Close"
    ) -> pd.DataFrame:
        """
        Calculate volatility indicators.
        
        Args:
            data: DataFrame with price data
            price_column: Column name for prices
            
        Returns:
            DataFrame with volatility indicators added
        """
        df = data.copy()
        prices = df[price_column]
        
        # Bollinger Bands
        for window in [10, 20, 30]:
            sma = prices.rolling(window).mean()
            std = prices.rolling(window).std()
            
            df[f"bb_upper_{window}"] = sma + (std * 2)
            df[f"bb_lower_{window}"] = sma - (std * 2)
            df[f"bb_middle_{window}"] = sma
            df[f"bb_width_{window}"] = (df[f"bb_upper_{window}"] - df[f"bb_lower_{window}"]) / sma
            df[f"bb_position_{window}"] = (prices - df[f"bb_lower_{window}"]) / (
                df[f"bb_upper_{window}"] - df[f"bb_lower_{window}"]
            )
        
        # Average True Range (ATR)
        if all(col in df.columns for col in ["High", "Low"]):
            df["atr_14"] = self._calculate_atr(df["High"], df["Low"], prices, 14)
            df["atr_20"] = self._calculate_atr(df["High"], df["Low"], prices, 20)
        
        # Historical Volatility
        returns = prices.pct_change()
        for window in [10, 20, 30, 60]:
            df[f"volatility_{window}"] = returns.rolling(window).std() * np.sqrt(252)
        
        # Keltner Channels
        if all(col in df.columns for col in ["High", "Low"]):
            ema_20 = prices.ewm(span=20).mean()
            atr_10 = self._calculate_atr(df["High"], df["Low"], prices, 10)
            
            df["keltner_upper"] = ema_20 + (atr_10 * 2)
            df["keltner_lower"] = ema_20 - (atr_10 * 2)
            df["keltner_middle"] = ema_20
        
        # Volatility Index (VIX-like calculation)
        df["volatility_index"] = self._calculate_volatility_index(prices)
        
        return df
    
    def calculate_trend_indicators(
        self,
        data: pd.DataFrame,
        price_column: str = "Close"
    ) -> pd.DataFrame:
        """
        Calculate trend indicators.
        
        Args:
            data: DataFrame with price data
            price_column: Column name for prices
            
        Returns:
            DataFrame with trend indicators added
        """
        df = data.copy()
        prices = df[price_column]
        
        # Average Directional Index (ADX)
        if all(col in df.columns for col in ["High", "Low"]):
            df["adx"] = self._calculate_adx(df["High"], df["Low"], prices)
        
        # Parabolic SAR
        if all(col in df.columns for col in ["High", "Low"]):
            df["parabolic_sar"] = self._calculate_parabolic_sar(df["High"], df["Low"])
        
        # Aroon Oscillator
        if all(col in df.columns for col in ["High", "Low"]):
            df["aroon_up"], df["aroon_down"] = self._calculate_aroon(df["High"], df["Low"])
            df["aroon_oscillator"] = df["aroon_up"] - df["aroon_down"]
        
        # Linear Regression Slope
        for window in [10, 20, 50]:
            df[f"lr_slope_{window}"] = self._calculate_linear_regression_slope(prices, window)
        
        # Trend Strength Index
        df["trend_strength"] = self._calculate_trend_strength(prices)
        
        # Ichimoku Cloud components
        if all(col in df.columns for col in ["High", "Low"]):
            ichimoku = self._calculate_ichimoku(df["High"], df["Low"], prices)
            for key, value in ichimoku.items():
                df[f"ichimoku_{key}"] = value
        
        return df
    
    def calculate_volume_indicators(
        self,
        data: pd.DataFrame,
        price_column: str = "Close",
        volume_column: str = "Volume"
    ) -> pd.DataFrame:
        """
        Calculate volume indicators.
        
        Args:
            data: DataFrame with price data
            price_column: Column name for prices
            volume_column: Column name for volume
            
        Returns:
            DataFrame with volume indicators added
        """
        if volume_column not in data.columns:
            logger.warning("Volume data not available, skipping volume indicators")
            return data
        
        df = data.copy()
        prices = df[price_column]
        volume = df[volume_column]
        
        # On-Balance Volume (OBV)
        df["obv"] = self._calculate_obv(prices, volume)
        
        # Volume Weighted Average Price (VWAP)
        if all(col in df.columns for col in ["High", "Low"]):
            df["vwap"] = self._calculate_vwap(df["High"], df["Low"], prices, volume)
        
        # Accumulation/Distribution Line
        if all(col in df.columns for col in ["High", "Low"]):
            df["ad_line"] = self._calculate_ad_line(df["High"], df["Low"], prices, volume)
        
        # Chaikin Money Flow
        if all(col in df.columns for col in ["High", "Low"]):
            df["cmf"] = self._calculate_cmf(df["High"], df["Low"], prices, volume)
        
        # Volume Rate of Change
        df["volume_roc"] = volume.pct_change(periods=10)
        
        # Volume Moving Averages
        df["volume_sma_20"] = volume.rolling(20).mean()
        df["volume_ema_20"] = volume.ewm(span=20).mean()
        
        # Volume Oscillator
        df["volume_oscillator"] = (
            (volume.rolling(5).mean() - volume.rolling(20).mean()) / 
            volume.rolling(20).mean() * 100
        )
        
        return df
    
    def detect_chart_patterns(
        self,
        data: pd.DataFrame,
        price_column: str = "Close"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect common chart patterns.
        
        Args:
            data: DataFrame with price data
            price_column: Column name for prices
            
        Returns:
            Dictionary with detected patterns
        """
        patterns = {
            "support_resistance": [],
            "head_shoulders": [],
            "double_top_bottom": [],
            "triangles": [],
            "flags_pennants": []
        }
        
        if len(data) < 50:
            return patterns
        
        prices = data[price_column].values
        
        # Support and Resistance levels
        patterns["support_resistance"] = self._detect_support_resistance(data, price_column)
        
        # Head and Shoulders patterns
        patterns["head_shoulders"] = self._detect_head_shoulders(prices)
        
        # Double Top/Bottom patterns
        patterns["double_top_bottom"] = self._detect_double_top_bottom(prices)
        
        # Triangle patterns
        if all(col in data.columns for col in ["High", "Low"]):
            patterns["triangles"] = self._detect_triangles(data)
        
        # Flag and Pennant patterns
        patterns["flags_pennants"] = self._detect_flags_pennants(prices)
        
        return patterns
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    
    def _calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_window: int = 14,
        d_window: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return k_percent, d_percent
    
    def _calculate_williams_r(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    def _calculate_roc(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Rate of Change."""
        return ((prices - prices.shift(window)) / prices.shift(window)) * 100
    
    def _calculate_mfi(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=window).sum()
        negative_mf = negative_flow.rolling(window=window).sum()
        
        mfr = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + mfr))
        
        return mfi
    
    def _calculate_cci(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """Calculate Commodity Channel Index."""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        
        # Mean deviation
        mad = typical_price.rolling(window=window).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    def _calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    def _calculate_hull_ma(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Hull Moving Average."""
        half_window = int(window / 2)
        sqrt_window = int(np.sqrt(window))
        
        wma_half = prices.rolling(half_window).apply(
            lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(),
            raw=True
        )
        wma_full = prices.rolling(window).apply(
            lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(),
            raw=True
        )
        
        diff = 2 * wma_half - wma_full
        hma = diff.rolling(sqrt_window).apply(
            lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(),
            raw=True
        )
        
        return hma
    
    def _calculate_volatility_index(self, prices: pd.Series, window: int = 30) -> pd.Series:
        """Calculate a VIX-like volatility index."""
        returns = prices.pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252) * 100
        
        # Smooth the volatility
        smoothed_volatility = volatility.ewm(span=10).mean()
        
        return smoothed_volatility
    
    def _calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """Calculate Average Directional Index."""
        # Calculate True Range
        tr = self._calculate_atr(high, low, close, 1)
        
        # Calculate Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        # Smooth the values
        tr_smooth = tr.rolling(window=window).mean()
        dm_plus_smooth = dm_plus.rolling(window=window).mean()
        dm_minus_smooth = dm_minus.rolling(window=window).mean()
        
        # Calculate Directional Indicators
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        
        # Calculate ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=window).mean()
        
        return adx
    
    def _calculate_parabolic_sar(
        self,
        high: pd.Series,
        low: pd.Series,
        acceleration: float = 0.02,
        maximum: float = 0.2
    ) -> pd.Series:
        """Calculate Parabolic SAR."""
        length = len(high)
        sar = np.zeros(length)
        trend = np.zeros(length)
        af = np.zeros(length)
        ep = np.zeros(length)
        
        # Initialize
        sar[0] = low.iloc[0]
        trend[0] = 1  # 1 for uptrend, -1 for downtrend
        af[0] = acceleration
        ep[0] = high.iloc[0]
        
        for i in range(1, length):
            # Calculate SAR
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            
            # Check for trend reversal
            if trend[i-1] == 1:  # Uptrend
                if low.iloc[i] <= sar[i]:
                    # Trend reversal to downtrend
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    af[i] = acceleration
                    ep[i] = low.iloc[i]
                else:
                    trend[i] = 1
                    if high.iloc[i] > ep[i-1]:
                        ep[i] = high.iloc[i]
                        af[i] = min(af[i-1] + acceleration, maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                    
                    # Ensure SAR doesn't exceed recent lows
                    sar[i] = min(sar[i], low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])
            
            else:  # Downtrend
                if high.iloc[i] >= sar[i]:
                    # Trend reversal to uptrend
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    af[i] = acceleration
                    ep[i] = high.iloc[i]
                else:
                    trend[i] = -1
                    if low.iloc[i] < ep[i-1]:
                        ep[i] = low.iloc[i]
                        af[i] = min(af[i-1] + acceleration, maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                    
                    # Ensure SAR doesn't fall below recent highs
                    sar[i] = max(sar[i], high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])
        
        return pd.Series(sar, index=high.index)
    
    def _calculate_aroon(
        self,
        high: pd.Series,
        low: pd.Series,
        window: int = 14
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Aroon indicators."""
        aroon_up = high.rolling(window=window).apply(
            lambda x: ((window - 1 - np.argmax(x)) / (window - 1)) * 100, raw=True
        )
        aroon_down = low.rolling(window=window).apply(
            lambda x: ((window - 1 - np.argmin(x)) / (window - 1)) * 100, raw=True
        )
        
        return aroon_up, aroon_down
    
    def _calculate_linear_regression_slope(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate linear regression slope."""
        def slope(y):
            x = np.arange(len(y))
            slope_val = stats.linregress(x, y)[0]
            return slope_val
        
        return prices.rolling(window=window).apply(slope, raw=True)
    
    def _calculate_trend_strength(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate trend strength indicator."""
        returns = prices.pct_change()
        
        # Calculate the ratio of positive to negative returns
        positive_returns = returns.where(returns > 0, 0)
        negative_returns = returns.where(returns < 0, 0)
        
        positive_sum = positive_returns.rolling(window=window).sum()
        negative_sum = abs(negative_returns.rolling(window=window).sum())
        
        trend_strength = (positive_sum - negative_sum) / (positive_sum + negative_sum)
        
        return trend_strength
    
    def _calculate_ichimoku(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> Dict[str, pd.Series]:
        """Calculate Ichimoku Cloud components."""
        # Tenkan-sen (Conversion Line): 9-period
        tenkan_sen = (high.rolling(9).max() + low.rolling(9).min()) / 2
        
        # Kijun-sen (Base Line): 26-period
        kijun_sen = (high.rolling(26).max() + low.rolling(26).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): 52-period
        senkou_span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-26)
        
        return {
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen,
            "senkou_span_a": senkou_span_a,
            "senkou_span_b": senkou_span_b,
            "chikou_span": chikou_span
        }
    
    def _calculate_obv(self, prices: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        price_change = prices.diff()
        obv = volume.copy()
        
        obv.loc[price_change < 0] = -volume
        obv.loc[price_change == 0] = 0
        obv = obv.cumsum()
        
        return obv
    
    def _calculate_vwap(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        return vwap
    
    def _calculate_ad_line(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # Handle division by zero
        
        ad_line = (clv * volume).cumsum()
        
        return ad_line
    
    def _calculate_cmf(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """Calculate Chaikin Money Flow."""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        
        cmf = (clv * volume).rolling(window=window).sum() / volume.rolling(window=window).sum()
        
        return cmf
    
    def _detect_support_resistance(
        self,
        data: pd.DataFrame,
        price_column: str,
        window: int = 20,
        min_touches: int = 2
    ) -> List[Dict[str, Any]]:
        """Detect support and resistance levels."""
        prices = data[price_column].values
        levels = []
        
        # Find local maxima and minima
        peaks, _ = find_peaks(prices, distance=window)
        troughs, _ = find_peaks(-prices, distance=window)
        
        # Combine and sort all levels
        all_levels = list(prices[peaks]) + list(prices[troughs])
        level_types = ['resistance'] * len(peaks) + ['support'] * len(troughs)
        
        # Group similar levels
        tolerance = np.std(prices) * 0.5
        grouped_levels = []
        
        for i, level in enumerate(all_levels):
            similar_levels = [
                j for j, other_level in enumerate(all_levels)
                if abs(level - other_level) <= tolerance
            ]
            
            if len(similar_levels) >= min_touches:
                avg_level = np.mean([all_levels[j] for j in similar_levels])
                level_type = max([level_types[j] for j in similar_levels], 
                               key=lambda x: [level_types[j] for j in similar_levels].count(x))
                
                grouped_levels.append({
                    'level': avg_level,
                    'type': level_type,
                    'touches': len(similar_levels),
                    'strength': len(similar_levels) / len(all_levels)
                })
        
        # Remove duplicates
        unique_levels = []
        for level in grouped_levels:
            if not any(abs(level['level'] - ul['level']) <= tolerance for ul in unique_levels):
                unique_levels.append(level)
        
        return unique_levels
    
    def _detect_head_shoulders(self, prices: np.ndarray, min_prominence: float = 0.02) -> List[Dict[str, Any]]:
        """Detect Head and Shoulders patterns."""
        patterns = []
        
        # Find peaks
        peaks, properties = find_peaks(prices, prominence=np.std(prices) * min_prominence)
        
        if len(peaks) < 3:
            return patterns
        
        # Look for head and shoulders pattern (3 consecutive peaks where middle is highest)
        for i in range(len(peaks) - 2):
            left_shoulder = peaks[i]
            head = peaks[i + 1]
            right_shoulder = peaks[i + 2]
            
            left_height = prices[left_shoulder]
            head_height = prices[head]
            right_height = prices[right_shoulder]
            
            # Check if it's a valid head and shoulders
            if (head_height > left_height and head_height > right_height and
                abs(left_height - right_height) / head_height < 0.05):  # Shoulders roughly equal
                
                # Find neckline (troughs between peaks)
                left_trough_idx = np.argmin(prices[left_shoulder:head]) + left_shoulder
                right_trough_idx = np.argmin(prices[head:right_shoulder]) + head
                
                neckline_level = (prices[left_trough_idx] + prices[right_trough_idx]) / 2
                
                patterns.append({
                    'type': 'head_and_shoulders',
                    'left_shoulder': {'index': left_shoulder, 'price': left_height},
                    'head': {'index': head, 'price': head_height},
                    'right_shoulder': {'index': right_shoulder, 'price': right_height},
                    'neckline': neckline_level,
                    'target': neckline_level - (head_height - neckline_level),
                    'confidence': min(1.0, (head_height - max(left_height, right_height)) / head_height * 10)
                })
        
        return patterns
    
    def _detect_double_top_bottom(self, prices: np.ndarray) -> List[Dict[str, Any]]:
        """Detect Double Top and Double Bottom patterns."""
        patterns = []
        
        # Find peaks and troughs
        peaks, _ = find_peaks(prices, prominence=np.std(prices) * 0.02)
        troughs, _ = find_peaks(-prices, prominence=np.std(prices) * 0.02)
        
        # Double Top detection
        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                peak1_idx, peak2_idx = peaks[i], peaks[j]
                peak1_price, peak2_price = prices[peak1_idx], prices[peak2_idx]
                
                # Check if peaks are roughly equal (within 2%)
                if abs(peak1_price - peak2_price) / max(peak1_price, peak2_price) < 0.02:
                    # Find trough between peaks
                    trough_idx = np.argmin(prices[peak1_idx:peak2_idx]) + peak1_idx
                    trough_price = prices[trough_idx]
                    
                    # Ensure significant decline between peaks
                    if (max(peak1_price, peak2_price) - trough_price) / max(peak1_price, peak2_price) > 0.05:
                        patterns.append({
                            'type': 'double_top',
                            'peak1': {'index': peak1_idx, 'price': peak1_price},
                            'peak2': {'index': peak2_idx, 'price': peak2_price},
                            'trough': {'index': trough_idx, 'price': trough_price},
                            'target': trough_price - (max(peak1_price, peak2_price) - trough_price),
                            'confidence': 1 - abs(peak1_price - peak2_price) / max(peak1_price, peak2_price)
                        })
        
        # Double Bottom detection (similar logic with troughs)
        for i in range(len(troughs) - 1):
            for j in range(i + 1, len(troughs)):
                trough1_idx, trough2_idx = troughs[i], troughs[j]
                trough1_price, trough2_price = prices[trough1_idx], prices[trough2_idx]
                
                if abs(trough1_price - trough2_price) / min(trough1_price, trough2_price) < 0.02:
                    peak_idx = np.argmax(prices[trough1_idx:trough2_idx]) + trough1_idx
                    peak_price = prices[peak_idx]
                    
                    if (peak_price - min(trough1_price, trough2_price)) / peak_price > 0.05:
                        patterns.append({
                            'type': 'double_bottom',
                            'trough1': {'index': trough1_idx, 'price': trough1_price},
                            'trough2': {'index': trough2_idx, 'price': trough2_price},
                            'peak': {'index': peak_idx, 'price': peak_price},
                            'target': peak_price + (peak_price - min(trough1_price, trough2_price)),
                            'confidence': 1 - abs(trough1_price - trough2_price) / min(trough1_price, trough2_price)
                        })
        
        return patterns
    
    def _detect_triangles(self, data: pd.DataFrame, min_length: int = 20) -> List[Dict[str, Any]]:
        """Detect triangle patterns."""
        patterns = []
        
        if len(data) < min_length * 2:
            return patterns
        
        high_prices = data['High'].values
        low_prices = data['Low'].values
        
        # Look for converging trend lines
        for start_idx in range(len(data) - min_length * 2):
            end_idx = start_idx + min_length * 2
            
            # Find peaks and troughs in the window
            window_high = high_prices[start_idx:end_idx]
            window_low = low_prices[start_idx:end_idx]
            
            peaks, _ = find_peaks(window_high, distance=5)
            troughs, _ = find_peaks(-window_low, distance=5)
            
            if len(peaks) >= 2 and len(troughs) >= 2:
                # Calculate trend lines
                peak_trend = np.polyfit(peaks, window_high[peaks], 1)
                trough_trend = np.polyfit(troughs, window_low[troughs], 1)
                
                # Check for convergence
                if abs(peak_trend[0]) > 0.001 or abs(trough_trend[0]) > 0.001:  # Not horizontal
                    if peak_trend[0] * trough_trend[0] < 0:  # Converging
                        # Determine triangle type
                        if peak_trend[0] < 0 and trough_trend[0] > 0:
                            triangle_type = "symmetrical"
                        elif peak_trend[0] < 0 and abs(trough_trend[0]) < 0.001:
                            triangle_type = "descending"
                        elif abs(peak_trend[0]) < 0.001 and trough_trend[0] > 0:
                            triangle_type = "ascending"
                        else:
                            triangle_type = "unknown"
                        
                        patterns.append({
                            'type': triangle_type + '_triangle',
                            'start_index': start_idx,
                            'end_index': end_idx,
                            'upper_trend_slope': peak_trend[0],
                            'lower_trend_slope': trough_trend[0],
                            'convergence_point': -((peak_trend[1] - trough_trend[1]) / 
                                                 (peak_trend[0] - trough_trend[0])),
                            'confidence': min(1.0, len(peaks) * len(troughs) / 10)
                        })
        
        return patterns
    
    def _detect_flags_pennants(self, prices: np.ndarray, min_trend_length: int = 10) -> List[Dict[str, Any]]:
        """Detect flag and pennant patterns."""
        patterns = []
        
        if len(prices) < min_trend_length * 3:
            return patterns
        
        # Calculate price changes
        price_changes = np.diff(prices)
        
        # Look for strong trends followed by consolidation
        for i in range(min_trend_length, len(prices) - min_trend_length * 2):
            # Check for strong trend before
            trend_before = prices[i] - prices[i - min_trend_length]
            trend_strength = abs(trend_before) / prices[i - min_trend_length]
            
            if trend_strength > 0.05:  # Significant trend
                # Look for consolidation after
                consolidation_period = prices[i:i + min_trend_length]
                consolidation_volatility = np.std(consolidation_period) / np.mean(consolidation_period)
                
                if consolidation_volatility < 0.02:  # Low volatility consolidation
                    # Determine pattern type
                    if trend_before > 0:
                        pattern_type = "bull_flag" if len(consolidation_period) > 5 else "bull_pennant"
                    else:
                        pattern_type = "bear_flag" if len(consolidation_period) > 5 else "bear_pennant"
                    
                    # Calculate target
                    flag_height = abs(trend_before)
                    if trend_before > 0:
                        target = prices[i + min_trend_length - 1] + flag_height
                    else:
                        target = prices[i + min_trend_length - 1] - flag_height
                    
                    patterns.append({
                        'type': pattern_type,
                        'trend_start': i - min_trend_length,
                        'flag_start': i,
                        'flag_end': i + min_trend_length,
                        'trend_strength': trend_strength,
                        'target': target,
                        'confidence': min(1.0, trend_strength * 10)
                    })
        
        return patterns