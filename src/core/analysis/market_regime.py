"""Market regime detection and analysis."""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.hidden_markov import GaussianHMM

from ...utils.exceptions import AnalysisError

logger = logging.getLogger(__name__)


class MarketRegimeAnalyzer:
    """Market regime detection and analysis engine."""
    
    def __init__(self, n_regimes: int = 3):
        """
        Initialize the market regime analyzer.
        
        Args:
            n_regimes: Number of market regimes to identify
        """
        self.n_regimes = n_regimes
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.scaler = StandardScaler()
    
    def detect_market_regimes(
        self,
        fund_data: Dict[str, pd.DataFrame],
        economic_data: Optional[Dict[str, Any]] = None,
        method: str = "hmm"
    ) -> Dict[str, Any]:
        """
        Detect market regimes using various methodologies.
        
        Args:
            fund_data: Dictionary of fund DataFrames
            economic_data: Optional economic data for regime detection
            method: Detection method ('hmm', 'gmm', 'kmeans', 'threshold')
            
        Returns:
            Dictionary with regime detection results
        """
        try:
            self.logger.info(f"Detecting market regimes using {method} method")
            
            # Prepare market indicators
            market_indicators = self._prepare_market_indicators(fund_data, economic_data)
            
            if market_indicators is None or market_indicators.empty:
                return {"error": "Insufficient data for regime detection"}
            
            # Apply selected detection method
            if method == "hmm":
                regime_results = self._detect_regimes_hmm(market_indicators)
            elif method == "gmm":
                regime_results = self._detect_regimes_gmm(market_indicators)
            elif method == "kmeans":
                regime_results = self._detect_regimes_kmeans(market_indicators)
            elif method == "threshold":
                regime_results = self._detect_regimes_threshold(market_indicators)
            else:
                raise ValueError(f"Unknown regime detection method: {method}")
            
            # Analyze regime characteristics
            regime_analysis = self._analyze_regime_characteristics(
                market_indicators, regime_results["regime_labels"]
            )
            
            # Calculate regime transition probabilities
            transition_probs = self._calculate_transition_probabilities(regime_results["regime_labels"])
            
            # Regime persistence analysis
            persistence_analysis = self._analyze_regime_persistence(regime_results["regime_labels"])
            
            # Fund performance by regime
            fund_performance_by_regime = self._analyze_fund_performance_by_regime(
                fund_data, regime_results["regime_labels"]
            )
            
            results = {
                "detection_method": method,
                "regime_results": regime_results,
                "regime_characteristics": regime_analysis,
                "transition_probabilities": transition_probs,
                "persistence_analysis": persistence_analysis,
                "fund_performance_by_regime": fund_performance_by_regime,
                "current_regime": self._identify_current_regime(regime_results, market_indicators),
                "regime_forecast": self._forecast_regime_changes(regime_results, market_indicators)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in market regime detection: {e}")
            raise AnalysisError(f"Market regime detection failed: {e}")
    
    def _prepare_market_indicators(
        self,
        fund_data: Dict[str, pd.DataFrame],
        economic_data: Optional[Dict[str, Any]] = None
    ) -> Optional[pd.DataFrame]:
        """Prepare market indicators for regime detection."""
        
        try:
            # Calculate market-wide indicators
            indicators = {}
            
            # 1. Market returns and volatility
            market_returns = self._calculate_market_returns(fund_data)
            if market_returns is not None:
                indicators['market_return'] = market_returns
                indicators['market_volatility'] = market_returns.rolling(20).std()
                indicators['market_volume'] = self._calculate_market_volume(fund_data)
            
            # 2. Technical indicators
            tech_indicators = self._calculate_technical_indicators(market_returns)
            indicators.update(tech_indicators)
            
            # 3. Cross-sectional dispersion
            dispersion_metrics = self._calculate_dispersion_metrics(fund_data)
            indicators.update(dispersion_metrics)
            
            # 4. Economic indicators (if available)
            if economic_data:
                econ_indicators = self._extract_economic_regime_indicators(economic_data)
                indicators.update(econ_indicators)
            
            # Combine all indicators
            if not indicators:
                return None
            
            indicators_df = pd.DataFrame(indicators)
            indicators_df = indicators_df.dropna()
            
            # Calculate additional derived indicators
            if 'market_return' in indicators_df.columns:
                # Rolling Sharpe ratio
                indicators_df['rolling_sharpe'] = (
                    indicators_df['market_return'].rolling(60).mean() * 252
                ) / (indicators_df['market_return'].rolling(60).std() * np.sqrt(252))
                
                # Trend indicators
                indicators_df['trend_20d'] = indicators_df['market_return'].rolling(20).sum()
                indicators_df['trend_60d'] = indicators_df['market_return'].rolling(60).sum()
                
                # Momentum indicators
                indicators_df['momentum'] = indicators_df['market_return'].rolling(10).sum()
            
            return indicators_df.dropna()
            
        except Exception as e:
            self.logger.error(f"Error preparing market indicators: {e}")
            return None
    
    def _calculate_market_returns(self, fund_data: Dict[str, pd.DataFrame]) -> Optional[pd.Series]:
        """Calculate market-wide returns (equal-weighted average)."""
        
        try:
            returns_list = []
            
            for fund_name, data in fund_data.items():
                if 'daily_return' in data.columns:
                    returns = data['daily_return'].dropna()
                    returns_list.append(returns)
            
            if not returns_list:
                return None
            
            # Align all return series and calculate equal-weighted market return
            aligned_returns = pd.concat(returns_list, axis=1, join='inner')
            market_returns = aligned_returns.mean(axis=1)
            
            return market_returns
            
        except Exception as e:
            self.logger.error(f"Error calculating market returns: {e}")
            return None
    
    def _calculate_market_volume(self, fund_data: Dict[str, pd.DataFrame]) -> pd.Series:
        """Calculate aggregate market volume indicator."""
        
        try:
            volume_list = []
            
            for fund_name, data in fund_data.items():
                if 'Volume' in data.columns:
                    volume = data['Volume'].dropna()
                    # Normalize volume for each fund
                    normalized_volume = volume / volume.rolling(252).mean()
                    volume_list.append(normalized_volume)
            
            if not volume_list:
                return pd.Series(dtype=float)
            
            # Align all volume series and calculate average
            aligned_volumes = pd.concat(volume_list, axis=1, join='inner')
            market_volume = aligned_volumes.mean(axis=1)
            
            return market_volume
            
        except Exception as e:
            self.logger.warning(f"Error calculating market volume: {e}")
            return pd.Series(dtype=float)
    
    def _calculate_technical_indicators(self, market_returns: pd.Series) -> Dict[str, pd.Series]:
        """Calculate technical indicators for regime detection."""
        
        if market_returns is None or market_returns.empty:
            return {}
        
        indicators = {}
        
        try:
            # Price level (cumulative returns)
            price_level = (1 + market_returns.fillna(0)).cumprod()
            
            # Moving averages
            indicators['sma_50'] = price_level.rolling(50).mean()
            indicators['sma_200'] = price_level.rolling(200).mean()
            indicators['price_ma_ratio'] = price_level / indicators['sma_200']
            
            # RSI
            indicators['rsi'] = self._calculate_rsi(market_returns)
            
            # MACD
            ema_12 = price_level.ewm(span=12).mean()
            ema_26 = price_level.ewm(span=26).mean()
            indicators['macd'] = ema_12 - ema_26
            indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
            
            # Bollinger Bands
            bb_middle = price_level.rolling(20).mean()
            bb_std = price_level.rolling(20).std()
            indicators['bb_position'] = (price_level - bb_middle) / (2 * bb_std)
            
            # VIX-like volatility indicator
            indicators['vol_indicator'] = market_returns.rolling(20).std() * np.sqrt(252)
            
        except Exception as e:
            self.logger.warning(f"Error calculating technical indicators: {e}")
        
        return indicators
    
    def _calculate_rsi(self, returns: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI from returns."""
        
        try:
            # Convert returns to price changes
            delta = returns
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series(dtype=float)
    
    def _calculate_dispersion_metrics(self, fund_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """Calculate cross-sectional dispersion metrics."""
        
        indicators = {}
        
        try:
            # Collect all daily returns
            returns_list = []
            for fund_name, data in fund_data.items():
                if 'daily_return' in data.columns:
                    returns_list.append(data['daily_return'])
            
            if len(returns_list) < 2:
                return indicators
            
            # Align returns
            aligned_returns = pd.concat(returns_list, axis=1, join='inner')
            
            # Cross-sectional standard deviation
            indicators['cross_sectional_dispersion'] = aligned_returns.std(axis=1)
            
            # Range of returns
            indicators['return_range'] = aligned_returns.max(axis=1) - aligned_returns.min(axis=1)
            
            # Percentage of funds with positive returns
            indicators['positive_return_pct'] = (aligned_returns > 0).mean(axis=1)
            
            # Correlation stability
            rolling_corr = aligned_returns.rolling(60).corr().groupby(level=0).mean().mean(axis=1)
            indicators['avg_correlation'] = rolling_corr
            
        except Exception as e:
            self.logger.warning(f"Error calculating dispersion metrics: {e}")
        
        return indicators
    
    def _extract_economic_regime_indicators(self, economic_data: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Extract economic indicators relevant for regime detection."""
        
        indicators = {}
        
        try:
            # Interest rate indicators
            if "repo_rate" in economic_data:
                repo_data = economic_data["repo_rate"].get("data", [])
                if repo_data:
                    df = pd.DataFrame(repo_data)
                    if "date" in df.columns and "value" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                        df = df.set_index("date")
                        indicators["repo_rate"] = df["value"]
                        indicators["repo_rate_change"] = df["value"].diff()
            
            # Yield curve slope
            if "government_bonds" in economic_data:
                bond_2y = None
                bond_10y = None
                
                for maturity, bond_data in economic_data["government_bonds"].items():
                    bond_series = bond_data.get("data", [])
                    if bond_series:
                        df = pd.DataFrame(bond_series)
                        if "date" in df.columns and "value" in df.columns:
                            df["date"] = pd.to_datetime(df["date"])
                            df = df.set_index("date")
                            
                            if maturity == "2y":
                                bond_2y = df["value"]
                            elif maturity == "10y":
                                bond_10y = df["value"]
                
                if bond_2y is not None and bond_10y is not None:
                    indicators["yield_curve_slope"] = bond_10y - bond_2y
            
            # Currency indicators
            if "exchange_rates" in economic_data:
                for currency, rate_data in economic_data["exchange_rates"].items():
                    rate_series = rate_data.get("data", [])
                    if rate_series:
                        df = pd.DataFrame(rate_series)
                        if "date" in df.columns and "value" in df.columns:
                            df["date"] = pd.to_datetime(df["date"])
                            df = df.set_index("date")
                            indicators[f"{currency.lower()}_rate"] = df["value"]
                            indicators[f"{currency.lower()}_volatility"] = df["value"].rolling(30).std()
            
        except Exception as e:
            self.logger.warning(f"Error extracting economic indicators: {e}")
        
        return indicators
    
    def _detect_regimes_hmm(self, indicators: pd.DataFrame) -> Dict[str, Any]:
        """Detect regimes using Hidden Markov Model."""
        
        try:
            # Prepare data
            data = indicators.select_dtypes(include=[np.number]).fillna(method='ffill').dropna()
            
            if data.empty or len(data) < 100:
                return {"error": "Insufficient data for HMM"}
            
            # Standardize data
            data_scaled = self.scaler.fit_transform(data)
            
            # Fit HMM
            model = GaussianHMM(n_components=self.n_regimes, covariance_type="full", random_state=42)
            model.fit(data_scaled)
            
            # Predict regimes
            regime_labels = model.predict(data_scaled)
            regime_probs = model.predict_proba(data_scaled)
            
            # Calculate model fit metrics
            log_likelihood = model.score(data_scaled)
            aic = -2 * log_likelihood + 2 * self._count_hmm_parameters(model)
            bic = -2 * log_likelihood + np.log(len(data)) * self._count_hmm_parameters(model)
            
            results = {
                "regime_labels": pd.Series(regime_labels, index=data.index),
                "regime_probabilities": pd.DataFrame(regime_probs, index=data.index, 
                                                   columns=[f"regime_{i}" for i in range(self.n_regimes)]),
                "transition_matrix": model.transmat_,
                "model_parameters": {
                    "means": model.means_,
                    "covariances": model.covars_,
                    "start_probabilities": model.startprob_
                },
                "model_fit": {
                    "log_likelihood": float(log_likelihood),
                    "aic": float(aic),
                    "bic": float(bic)
                }
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"HMM regime detection failed: {e}")
            return {"error": str(e)}
    
    def _count_hmm_parameters(self, model) -> int:
        """Count the number of parameters in HMM model."""
        
        n_states = model.n_components
        n_features = model.means_.shape[1]
        
        # Transition matrix parameters
        transition_params = n_states * (n_states - 1)
        
        # Initial state parameters
        initial_params = n_states - 1
        
        # Emission parameters (means + covariances)
        mean_params = n_states * n_features
        
        if model.covariance_type == "full":
            cov_params = n_states * n_features * (n_features + 1) // 2
        elif model.covariance_type == "diag":
            cov_params = n_states * n_features
        else:
            cov_params = n_states  # spherical or tied
        
        return transition_params + initial_params + mean_params + cov_params
    
    def _detect_regimes_gmm(self, indicators: pd.DataFrame) -> Dict[str, Any]:
        """Detect regimes using Gaussian Mixture Model."""
        
        try:
            # Prepare data
            data = indicators.select_dtypes(include=[np.number]).fillna(method='ffill').dropna()
            
            if data.empty:
                return {"error": "No valid data for GMM"}
            
            # Standardize data
            data_scaled = self.scaler.fit_transform(data)
            
            # Fit GMM
            gmm = GaussianMixture(n_components=self.n_regimes, random_state=42)
            gmm.fit(data_scaled)
            
            # Predict regimes
            regime_labels = gmm.predict(data_scaled)
            regime_probs = gmm.predict_proba(data_scaled)
            
            results = {
                "regime_labels": pd.Series(regime_labels, index=data.index),
                "regime_probabilities": pd.DataFrame(regime_probs, index=data.index,
                                                   columns=[f"regime_{i}" for i in range(self.n_regimes)]),
                "model_parameters": {
                    "means": gmm.means_,
                    "covariances": gmm.covariances_,
                    "weights": gmm.weights_
                },
                "model_fit": {
                    "aic": float(gmm.aic(data_scaled)),
                    "bic": float(gmm.bic(data_scaled)),
                    "log_likelihood": float(gmm.score(data_scaled))
                }
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"GMM regime detection failed: {e}")
            return {"error": str(e)}
    
    def _detect_regimes_kmeans(self, indicators: pd.DataFrame) -> Dict[str, Any]:
        """Detect regimes using K-means clustering."""
        
        try:
            # Prepare data
            data = indicators.select_dtypes(include=[np.number]).fillna(method='ffill').dropna()
            
            if data.empty:
                return {"error": "No valid data for K-means"}
            
            # Standardize data
            data_scaled = self.scaler.fit_transform(data)
            
            # Fit K-means
            kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
            regime_labels = kmeans.fit_predict(data_scaled)
            
            # Calculate distances to centroids as "probabilities"
            distances = kmeans.transform(data_scaled)
            max_distances = distances.max(axis=1, keepdims=True)
            regime_probs = 1 - (distances / max_distances)
            regime_probs = regime_probs / regime_probs.sum(axis=1, keepdims=True)
            
            results = {
                "regime_labels": pd.Series(regime_labels, index=data.index),
                "regime_probabilities": pd.DataFrame(regime_probs, index=data.index,
                                                   columns=[f"regime_{i}" for i in range(self.n_regimes)]),
                "cluster_centers": kmeans.cluster_centers_,
                "model_fit": {
                    "inertia": float(kmeans.inertia_),
                    "n_iter": int(kmeans.n_iter_)
                }
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"K-means regime detection failed: {e}")
            return {"error": str(e)}
    
    def _detect_regimes_threshold(self, indicators: pd.DataFrame) -> Dict[str, Any]:
        """Detect regimes using threshold-based rules."""
        
        try:
            if 'market_volatility' not in indicators.columns:
                return {"error": "Market volatility required for threshold method"}
            
            data = indicators[['market_volatility']].dropna()
            
            if data.empty:
                return {"error": "No volatility data available"}
            
            # Define volatility thresholds
            vol_25 = data['market_volatility'].quantile(0.25)
            vol_75 = data['market_volatility'].quantile(0.75)
            
            # Assign regimes based on volatility
            regime_labels = pd.Series(index=data.index, dtype=int)
            regime_labels[data['market_volatility'] <= vol_25] = 0  # Low volatility
            regime_labels[(data['market_volatility'] > vol_25) & (data['market_volatility'] <= vol_75)] = 1  # Medium
            regime_labels[data['market_volatility'] > vol_75] = 2  # High volatility
            
            # Create dummy probabilities
            regime_probs = pd.DataFrame(0.0, index=data.index, columns=[f"regime_{i}" for i in range(3)])
            for i in range(len(regime_labels)):
                regime_probs.iloc[i, regime_labels.iloc[i]] = 1.0
            
            results = {
                "regime_labels": regime_labels,
                "regime_probabilities": regime_probs,
                "thresholds": {
                    "low_vol_threshold": float(vol_25),
                    "high_vol_threshold": float(vol_75)
                },
                "regime_definitions": {
                    0: "Low Volatility",
                    1: "Medium Volatility", 
                    2: "High Volatility"
                }
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Threshold regime detection failed: {e}")
            return {"error": str(e)}
    
    def _analyze_regime_characteristics(
        self,
        indicators: pd.DataFrame,
        regime_labels: pd.Series
    ) -> Dict[str, Any]:
        """Analyze characteristics of each detected regime."""
        
        try:
            regime_characteristics = {}
            
            for regime_id in regime_labels.unique():
                regime_mask = regime_labels == regime_id
                regime_data = indicators[regime_mask]
                
                if regime_data.empty:
                    continue
                
                characteristics = {
                    "duration_stats": {
                        "total_periods": int(regime_mask.sum()),
                        "percentage_of_time": float(regime_mask.sum() / len(regime_labels) * 100)
                    },
                    "indicator_statistics": {}
                }
                
                # Calculate statistics for each indicator during this regime
                for column in indicators.select_dtypes(include=[np.number]).columns:
                    if column in regime_data.columns:
                        regime_values = regime_data[column].dropna()
                        if len(regime_values) > 0:
                            characteristics["indicator_statistics"][column] = {
                                "mean": float(regime_values.mean()),
                                "std": float(regime_values.std()),
                                "min": float(regime_values.min()),
                                "max": float(regime_values.max()),
                                "median": float(regime_values.median())
                            }
                
                # Identify defining characteristics
                characteristics["defining_features"] = self._identify_defining_features(
                    regime_data, indicators, regime_id
                )
                
                regime_characteristics[f"regime_{regime_id}"] = characteristics
            
            return regime_characteristics
            
        except Exception as e:
            self.logger.error(f"Error analyzing regime characteristics: {e}")
            return {"error": str(e)}
    
    def _identify_defining_features(
        self,
        regime_data: pd.DataFrame,
        all_data: pd.DataFrame,
        regime_id: int
    ) -> List[str]:
        """Identify the defining features of a regime."""
        
        defining_features = []
        
        try:
            for column in all_data.select_dtypes(include=[np.number]).columns:
                if column in regime_data.columns and len(regime_data[column].dropna()) > 0:
                    regime_mean = regime_data[column].mean()
                    overall_mean = all_data[column].mean()
                    
                    # Check if regime mean is significantly different
                    if abs(regime_mean - overall_mean) > all_data[column].std():
                        if regime_mean > overall_mean:
                            defining_features.append(f"High {column}")
                        else:
                            defining_features.append(f"Low {column}")
            
        except Exception as e:
            self.logger.debug(f"Error identifying defining features: {e}")
        
        return defining_features[:5]  # Top 5 defining features
    
    def _calculate_transition_probabilities(self, regime_labels: pd.Series) -> Dict[str, Any]:
        """Calculate regime transition probabilities."""
        
        try:
            unique_regimes = sorted(regime_labels.unique())
            n_regimes = len(unique_regimes)
            
            # Initialize transition matrix
            transition_matrix = np.zeros((n_regimes, n_regimes))
            
            # Count transitions
            for i in range(len(regime_labels) - 1):
                current_regime = regime_labels.iloc[i]
                next_regime = regime_labels.iloc[i + 1]
                
                current_idx = unique_regimes.index(current_regime)
                next_idx = unique_regimes.index(next_regime)
                
                transition_matrix[current_idx, next_idx] += 1
            
            # Normalize to get probabilities
            row_sums = transition_matrix.sum(axis=1)
            transition_probs = transition_matrix / row_sums[:, np.newaxis]
            transition_probs = np.nan_to_num(transition_probs)  # Handle division by zero
            
            # Create readable transition matrix
            transition_dict = {}
            for i, from_regime in enumerate(unique_regimes):
                transition_dict[f"from_regime_{from_regime}"] = {}
                for j, to_regime in enumerate(unique_regimes):
                    transition_dict[f"from_regime_{from_regime}"][f"to_regime_{to_regime}"] = float(transition_probs[i, j])
            
            # Calculate average regime duration
            regime_durations = {}
            for regime in unique_regimes:
                regime_mask = regime_labels == regime
                durations = []
                current_duration = 0
                
                for is_regime in regime_mask:
                    if is_regime:
                        current_duration += 1
                    else:
                        if current_duration > 0:
                            durations.append(current_duration)
                            current_duration = 0
                
                if current_duration > 0:
                    durations.append(current_duration)
                
                if durations:
                    regime_durations[f"regime_{regime}"] = {
                        "average_duration": float(np.mean(durations)),
                        "median_duration": float(np.median(durations)),
                        "max_duration": int(np.max(durations)),
                        "min_duration": int(np.min(durations))
                    }
            
            return {
                "transition_matrix": transition_dict,
                "regime_durations": regime_durations,
                "persistence_probabilities": {
                    f"regime_{regime}": float(transition_probs[i, i])
                    for i, regime in enumerate(unique_regimes)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating transition probabilities: {e}")
            return {"error": str(e)}
    
    def _analyze_regime_persistence(self, regime_labels: pd.Series) -> Dict[str, Any]:
        """Analyze regime persistence characteristics."""
        
        try:
            persistence_analysis = {}
            
            # Calculate regime switches
            regime_changes = (regime_labels != regime_labels.shift(1)).sum()
            total_periods = len(regime_labels)
            
            persistence_analysis["overall_stability"] = {
                "total_regime_switches": int(regime_changes),
                "switch_frequency": float(regime_changes / total_periods),
                "average_regime_length": float(total_periods / regime_changes) if regime_changes > 0 else float('inf')
            }
            
            # Analyze recent regime stability
            recent_periods = min(60, len(regime_labels))  # Last 60 periods or all data
            recent_labels = regime_labels.tail(recent_periods)
            recent_changes = (recent_labels != recent_labels.shift(1)).sum()
            
            persistence_analysis["recent_stability"] = {
                "recent_regime_switches": int(recent_changes),
                "recent_switch_frequency": float(recent_changes / recent_periods),
                "current_regime_duration": self._calculate_current_regime_duration(regime_labels)
            }
            
            return persistence_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing regime persistence: {e}")
            return {"error": str(e)}
    
    def _calculate_current_regime_duration(self, regime_labels: pd.Series) -> int:
        """Calculate how long the current regime has been active."""
        
        if len(regime_labels) == 0:
            return 0
        
        current_regime = regime_labels.iloc[-1]
        duration = 1
        
        for i in range(len(regime_labels) - 2, -1, -1):
            if regime_labels.iloc[i] == current_regime:
                duration += 1
            else:
                break
        
        return duration
    
    def _analyze_fund_performance_by_regime(
        self,
        fund_data: Dict[str, pd.DataFrame],
        regime_labels: pd.Series
    ) -> Dict[str, Any]:
        """Analyze fund performance in different market regimes."""
        
        try:
            performance_by_regime = {}
            
            for fund_name, data in fund_data.items():
                if 'daily_return' in data.columns:
                    fund_returns = data['daily_return']
                    
                    # Align with regime labels
                    aligned_data = pd.concat([fund_returns, regime_labels], axis=1, join='inner')
                    aligned_data.columns = ['returns', 'regime']
                    aligned_data = aligned_data.dropna()
                    
                    if aligned_data.empty:
                        continue
                    
                    fund_regime_performance = {}
                    
                    for regime in aligned_data['regime'].unique():
                        regime_returns = aligned_data[aligned_data['regime'] == regime]['returns']
                        
                        if len(regime_returns) > 0:
                            fund_regime_performance[f"regime_{regime}"] = {
                                "mean_return": float(regime_returns.mean()),
                                "volatility": float(regime_returns.std()),
                                "sharpe_ratio": float(regime_returns.mean() / regime_returns.std()) if regime_returns.std() > 0 else 0,
                                "positive_return_days": float((regime_returns > 0).mean()),
                                "max_return": float(regime_returns.max()),
                                "min_return": float(regime_returns.min()),
                                "periods": len(regime_returns)
                            }
                    
                    performance_by_regime[fund_name] = fund_regime_performance
            
            # Calculate regime rankings for each fund
            regime_rankings = self._calculate_regime_rankings(performance_by_regime)
            
            return {
                "individual_fund_performance": performance_by_regime,
                "regime_rankings": regime_rankings
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing fund performance by regime: {e}")
            return {"error": str(e)}
    
    def _calculate_regime_rankings(self, performance_by_regime: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate which regimes are best/worst for each fund."""
        
        try:
            rankings = {}
            
            for fund_name, regime_performance in performance_by_regime.items():
                if isinstance(regime_performance, dict):
                    # Rank regimes by mean return
                    regime_returns = []
                    for regime, metrics in regime_performance.items():
                        if isinstance(metrics, dict) and "mean_return" in metrics:
                            regime_returns.append((regime, metrics["mean_return"]))
                    
                    if regime_returns:
                        regime_returns.sort(key=lambda x: x[1], reverse=True)
                        
                        rankings[fund_name] = {
                            "best_regime": regime_returns[0][0] if regime_returns else None,
                            "worst_regime": regime_returns[-1][0] if regime_returns else None,
                            "return_spread": float(regime_returns[0][1] - regime_returns[-1][1]) if len(regime_returns) >= 2 else 0
                        }
            
            return rankings
            
        except Exception as e:
            self.logger.warning(f"Error calculating regime rankings: {e}")
            return {}
    
    def _identify_current_regime(
        self,
        regime_results: Dict[str, Any],
        market_indicators: pd.DataFrame
    ) -> Dict[str, Any]:
        """Identify the current market regime."""
        
        try:
            if "regime_labels" not in regime_results:
                return {"error": "No regime labels available"}
            
            regime_labels = regime_results["regime_labels"]
            
            if regime_labels.empty:
                return {"error": "No regime data available"}
            
            current_regime = regime_labels.iloc[-1]
            
            # Get current regime probability if available
            current_probability = None
            if "regime_probabilities" in regime_results:
                regime_probs = regime_results["regime_probabilities"]
                if not regime_probs.empty:
                    current_probability = float(regime_probs.iloc[-1, current_regime])
            
            # Calculate regime confidence based on recent stability
            recent_labels = regime_labels.tail(20)  # Last 20 periods
            regime_stability = (recent_labels == current_regime).mean()
            
            current_regime_info = {
                "current_regime": int(current_regime),
                "regime_probability": current_probability,
                "regime_stability": float(regime_stability),
                "regime_duration": self._calculate_current_regime_duration(regime_labels),
                "last_updated": regime_labels.index[-1].isoformat() if not regime_labels.index.empty else None
            }
            
            return current_regime_info
            
        except Exception as e:
            self.logger.error(f"Error identifying current regime: {e}")
            return {"error": str(e)}
    
    def _forecast_regime_changes(
        self,
        regime_results: Dict[str, Any],
        market_indicators: pd.DataFrame
    ) -> Dict[str, Any]:
        """Forecast potential regime changes."""
        
        try:
            forecast = {"method": "simple_heuristic"}
            
            if "regime_labels" not in regime_results:
                return {"error": "No regime labels for forecasting"}
            
            regime_labels = regime_results["regime_labels"]
            
            if len(regime_labels) < 20:
                return {"error": "Insufficient data for forecasting"}
            
            current_regime = regime_labels.iloc[-1]
            current_duration = self._calculate_current_regime_duration(regime_labels)
            
            # Simple heuristic forecasting based on:
            # 1. Current regime duration vs historical average
            # 2. Recent indicator trends
            # 3. Transition probabilities
            
            # Get transition probabilities
            transition_probs = None
            if "transition_matrix" in regime_results:
                from_current = f"from_regime_{current_regime}"
                if from_current in regime_results["transition_matrix"]:
                    transition_probs = regime_results["transition_matrix"][from_current]
            
            # Analyze recent indicator trends
            trend_analysis = self._analyze_recent_trends(market_indicators)
            
            # Calculate regime change probability
            regime_change_probability = self._estimate_regime_change_probability(
                current_duration, regime_labels, transition_probs
            )
            
            forecast.update({
                "current_regime": int(current_regime),
                "current_regime_duration": current_duration,
                "regime_change_probability": regime_change_probability,
                "trend_analysis": trend_analysis,
                "next_likely_regimes": self._identify_next_likely_regimes(transition_probs) if transition_probs else None
            })
            
            return forecast
            
        except Exception as e:
            self.logger.error(f"Error forecasting regime changes: {e}")
            return {"error": str(e)}
    
    def _analyze_recent_trends(self, market_indicators: pd.DataFrame) -> Dict[str, str]:
        """Analyze recent trends in market indicators."""
        
        trends = {}
        
        try:
            # Look at last 10 periods for trend
            recent_data = market_indicators.tail(10)
            
            for column in recent_data.select_dtypes(include=[np.number]).columns:
                series = recent_data[column].dropna()
                if len(series) >= 5:
                    # Simple trend calculation
                    first_half = series.iloc[:len(series)//2].mean()
                    second_half = series.iloc[len(series)//2:].mean()
                    
                    if second_half > first_half * 1.05:
                        trends[column] = "increasing"
                    elif second_half < first_half * 0.95:
                        trends[column] = "decreasing"
                    else:
                        trends[column] = "stable"
            
        except Exception as e:
            self.logger.debug(f"Error analyzing recent trends: {e}")
        
        return trends
    
    def _estimate_regime_change_probability(
        self,
        current_duration: int,
        regime_labels: pd.Series,
        transition_probs: Optional[Dict[str, float]]
    ) -> float:
        """Estimate probability of regime change."""
        
        try:
            # Base probability from transition matrix
            base_prob = 0.1  # Default 10% chance
            
            if transition_probs:
                # Probability of NOT staying in current regime
                current_regime = regime_labels.iloc[-1]
                stay_prob = transition_probs.get(f"to_regime_{current_regime}", 0.8)
                base_prob = 1 - stay_prob
            
            # Adjust based on duration (longer duration -> higher change probability)
            # Calculate historical average duration for current regime
            current_regime = regime_labels.iloc[-1]
            regime_durations = []
            current_streak = 0
            
            for i in range(len(regime_labels)):
                if regime_labels.iloc[i] == current_regime:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        regime_durations.append(current_streak)
                        current_streak = 0
            
            if regime_durations and len(regime_durations) > 1:  # Exclude current streak
                avg_duration = np.mean(regime_durations[:-1])  # Exclude current ongoing
                
                # Increase probability if current duration is longer than average
                if current_duration > avg_duration:
                    duration_factor = min(2.0, current_duration / avg_duration)
                    base_prob = min(0.8, base_prob * duration_factor)
            
            return float(base_prob)
            
        except Exception as e:
            self.logger.debug(f"Error estimating regime change probability: {e}")
            return 0.1
    
    def _identify_next_likely_regimes(self, transition_probs: Dict[str, float]) -> List[Dict[str, float]]:
        """Identify most likely next regimes."""
        
        try:
            # Convert transition probabilities to list of (regime, probability) tuples
            regime_probs = []
            for to_regime, prob in transition_probs.items():
                regime_id = to_regime.replace("to_regime_", "")
                regime_probs.append({"regime": regime_id, "probability": float(prob)})
            
            # Sort by probability
            regime_probs.sort(key=lambda x: x["probability"], reverse=True)
            
            return regime_probs[:3]  # Top 3 most likely next regimes
            
        except Exception as e:
            self.logger.debug(f"Error identifying next likely regimes: {e}")
            return []
    
    def generate_regime_report(
        self,
        fund_data: Dict[str, pd.DataFrame],
        economic_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive market regime analysis report."""
        
        try:
            self.logger.info("Generating comprehensive market regime report")
            
            # Run regime detection with multiple methods
            methods = ["hmm", "gmm", "threshold"]
            regime_results = {}
            
            for method in methods:
                try:
                    result = self.detect_market_regimes(fund_data, economic_data, method)
                    if "error" not in result:
                        regime_results[method] = result
                except Exception as e:
                    self.logger.warning(f"Regime detection with {method} failed: {e}")
            
            if not regime_results:
                return {"error": "All regime detection methods failed"}
            
            # Select best method based on model fit
            best_method = self._select_best_regime_method(regime_results)
            primary_results = regime_results[best_method]
            
            # Regime comparison across methods
            method_comparison = self._compare_regime_methods(regime_results)
            
            # Investment implications
            investment_implications = self._analyze_investment_implications(primary_results, fund_data)
            
            report = {
                "executive_summary": {
                    "primary_method": best_method,
                    "current_regime": primary_results.get("current_regime", {}),
                    "regime_stability": self._assess_regime_stability(primary_results),
                    "key_insights": self._generate_regime_insights(primary_results)
                },
                "detailed_analysis": {
                    "primary_results": primary_results,
                    "method_comparison": method_comparison,
                    "investment_implications": investment_implications
                },
                "alternative_methods": {k: v for k, v in regime_results.items() if k != best_method},
                "recommendations": self._generate_regime_recommendations(primary_results, fund_data),
                "report_metadata": {
                    "analysis_date": pd.Timestamp.now().isoformat(),
                    "methods_used": list(regime_results.keys()),
                    "data_coverage": self._assess_regime_data_coverage(fund_data)
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating regime report: {e}")
            raise AnalysisError(f"Regime report generation failed: {e}")
    
    def _select_best_regime_method(self, regime_results: Dict[str, Any]) -> str:
        """Select the best regime detection method based on model fit."""
        
        try:
            method_scores = {}
            
            for method, results in regime_results.items():
                score = 0
                
                # Check if we have model fit metrics
                if "regime_results" in results and "model_fit" in results["regime_results"]:
                    model_fit = results["regime_results"]["model_fit"]
                    
                    # For HMM and GMM, use BIC (lower is better)
                    if "bic" in model_fit:
                        score = -model_fit["bic"]  # Negative because lower BIC is better
                    elif "aic" in model_fit:
                        score = -model_fit["aic"]  # Negative because lower AIC is better
                    elif "log_likelihood" in model_fit:
                        score = model_fit["log_likelihood"]
                
                # Penalize methods with errors
                if "error" in results:
                    score = float('-inf')
                
                method_scores[method] = score
            
            # Return method with highest score
            if method_scores:
                best_method = max(method_scores, key=method_scores.get)
                return best_method
            
            # Fallback: prefer HMM > GMM > threshold > kmeans
            preference_order = ["hmm", "gmm", "threshold", "kmeans"]
            for method in preference_order:
                if method in regime_results:
                    return method
            
            # Last resort: return any available method
            return list(regime_results.keys())[0]
            
        except Exception as e:
            self.logger.warning(f"Error selecting best method: {e}")
            return list(regime_results.keys())[0] if regime_results else "hmm"
    
    def _compare_regime_methods(self, regime_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare regime detection results across methods."""
        
        comparison = {}
        
        try:
            # Compare number of regime switches
            switch_comparison = {}
            for method, results in regime_results.items():
                if "regime_results" in results and "regime_labels" in results["regime_results"]:
                    labels = results["regime_results"]["regime_labels"]
                    switches = (labels != labels.shift(1)).sum()
                    switch_comparison[method] = int(switches)
            
            comparison["regime_switches"] = switch_comparison
            
            # Compare current regime identification
            current_regime_comparison = {}
            for method, results in regime_results.items():
                if "current_regime" in results:
                    current_regime_comparison[method] = results["current_regime"]
            
            comparison["current_regime_identification"] = current_regime_comparison
            
            # Calculate agreement between methods
            if len(regime_results) >= 2:
                comparison["method_agreement"] = self._calculate_method_agreement(regime_results)
            
        except Exception as e:
            self.logger.warning(f"Error comparing regime methods: {e}")
        
        return comparison
    
    def _calculate_method_agreement(self, regime_results: Dict[str, Any]) -> float:
        """Calculate agreement between different regime detection methods."""
        
        try:
            # Get regime labels from all methods
            all_labels = {}
            for method, results in regime_results.items():
                if "regime_results" in results and "regime_labels" in results["regime_results"]:
                    all_labels[method] = results["regime_results"]["regime_labels"]
            
            if len(all_labels) < 2:
                return 1.0
            
            # Find common time periods
            common_index = None
            for labels in all_labels.values():
                if common_index is None:
                    common_index = labels.index
                else:
                    common_index = common_index.intersection(labels.index)
            
            if len(common_index) == 0:
                return 0.0
            
            # Align all label series
            aligned_labels = pd.DataFrame({
                method: labels.loc[common_index]
                for method, labels in all_labels.items()
            })
            
            # Calculate pairwise agreement
            methods = list(aligned_labels.columns)
            total_agreements = 0
            total_comparisons = 0
            
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    method1, method2 = methods[i], methods[j]
                    # Simple agreement: same regime changes at same time
                    changes1 = (aligned_labels[method1] != aligned_labels[method1].shift(1))
                    changes2 = (aligned_labels[method2] != aligned_labels[method2].shift(1))
                    
                    agreement = (changes1 == changes2).mean()
                    total_agreements += agreement
                    total_comparisons += 1
            
            return float(total_agreements / total_comparisons) if total_comparisons > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating method agreement: {e}")
            return 0.0
    
    def _assess_regime_stability(self, regime_results: Dict[str, Any]) -> str:
        """Assess overall regime stability."""
        
        try:
            if "persistence_analysis" not in regime_results:
                return "unknown"
            
            persistence = regime_results["persistence_analysis"]
            
            if "overall_stability" in persistence:
                switch_freq = persistence["overall_stability"].get("switch_frequency", 1.0)
                
                if switch_freq < 0.05:  # Less than 5% of periods see regime switch
                    return "very_stable"
                elif switch_freq < 0.1:
                    return "stable" 
                elif switch_freq < 0.2:
                    return "moderate"
                else:
                    return "volatile"
            
            return "unknown"
            
        except Exception as e:
            self.logger.debug(f"Error assessing regime stability: {e}")
            return "unknown"
    
    def _generate_regime_insights(self, regime_results: Dict[str, Any]) -> List[str]:
        """Generate key insights from regime analysis."""
        
        insights = []
        
        try:
            # Current regime insights
            if "current_regime" in regime_results:
                current = regime_results["current_regime"]
                if "current_regime" in current:
                    insights.append(f"Currently in regime {current['current_regime']}")
                    
                    if "regime_duration" in current:
                        duration = current["regime_duration"]
                        insights.append(f"Current regime has persisted for {duration} periods")
            
            # Stability insights
            stability = self._assess_regime_stability(regime_results)
            if stability != "unknown":
                insights.append(f"Market regime stability is {stability}")
            
            # Transition insights
            if "transition_probabilities" in regime_results:
                transitions = regime_results["transition_probabilities"]
                if "persistence_probabilities" in transitions:
                    max_persistence = max(transitions["persistence_probabilities"].values())
                    insights.append(f"Highest regime persistence probability: {max_persistence:.1%}")
            
        except Exception as e:
            self.logger.debug(f"Error generating regime insights: {e}")
        
        return insights[:5]  # Top 5 insights
    
    def _analyze_investment_implications(
        self,
        regime_results: Dict[str, Any],
        fund_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Analyze investment implications of current regime."""
        
        implications = {}
        
        try:
            # Get current regime
            current_regime_info = regime_results.get("current_regime", {})
            current_regime = current_regime_info.get("current_regime")
            
            if current_regime is None:
                return {"error": "Current regime not identified"}
            
            # Get fund performance in current regime
            fund_performance = regime_results.get("fund_performance_by_regime", {})
            
            if "individual_fund_performance" in fund_performance:
                individual_performance = fund_performance["individual_fund_performance"]
                
                # Identify best and worst performing funds in current regime
                regime_key = f"regime_{current_regime}"
                fund_returns_in_regime = []
                
                for fund_name, performance in individual_performance.items():
                    if regime_key in performance:
                        mean_return = performance[regime_key].get("mean_return", 0)
                        fund_returns_in_regime.append((fund_name, mean_return))
                
                if fund_returns_in_regime:
                    fund_returns_in_regime.sort(key=lambda x: x[1], reverse=True)
                    
                    implications["current_regime_performance"] = {
                        "best_performers": fund_returns_in_regime[:3],
                        "worst_performers": fund_returns_in_regime[-3:],
                        "performance_spread": fund_returns_in_regime[0][1] - fund_returns_in_regime[-1][1]
                    }
            
            # Regime change implications
            if "regime_forecast" in regime_results:
                forecast = regime_results["regime_forecast"]
                change_prob = forecast.get("regime_change_probability", 0)
                
                implications["regime_change_risk"] = {
                    "probability": change_prob,
                    "risk_level": "high" if change_prob > 0.3 else "medium" if change_prob > 0.15 else "low"
                }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing investment implications: {e}")
            implications["error"] = str(e)
        
        return implications
    
    def _generate_regime_recommendations(
        self,
        regime_results: Dict[str, Any], 
        fund_data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """Generate investment recommendations based on regime analysis."""
        
        recommendations = []
        
        try:
            # Current regime recommendations
            current_regime_info = regime_results.get("current_regime", {})
            current_regime = current_regime_info.get("current_regime")
            
            if current_regime is not None:
                # Get regime characteristics
                characteristics = regime_results.get("regime_characteristics", {})
                regime_key = f"regime_{current_regime}"
                
                if regime_key in characteristics:
                    defining_features = characteristics[regime_key].get("defining_features", [])
                    
                    for feature in defining_features[:2]:  # Top 2 features
                        if "High market_volatility" in feature:
                            recommendations.append("Consider defensive positioning due to high volatility regime")
                        elif "Low market_volatility" in feature:
                            recommendations.append("Current low volatility regime may favor growth strategies")
                        elif "High market_return" in feature:
                            recommendations.append("Positive momentum regime supports risk-on positioning")
            
            # Regime change recommendations
            if "regime_forecast" in regime_results:
                forecast = regime_results["regime_forecast"]
                change_prob = forecast.get("regime_change_probability", 0)
                
                if change_prob > 0.3:
                    recommendations.append("High regime change probability suggests tactical rebalancing")
                elif change_prob > 0.15:
                    recommendations.append("Monitor for regime change signals")
            
            # Fund-specific recommendations
            fund_performance = regime_results.get("fund_performance_by_regime", {})
            if "regime_rankings" in fund_performance:
                rankings = fund_performance["regime_rankings"]
                
                # Find funds that consistently perform well across regimes
                consistent_performers = []
                for fund_name, ranking in rankings.items():
                    if isinstance(ranking, dict) and ranking.get("return_spread", 0) < 0.001:  # Low spread = consistent
                        consistent_performers.append(fund_name)
                
                if consistent_performers:
                    recommendations.append(
                        f"Consider {consistent_performers[0]} for regime-independent performance"
                    )
            
        except Exception as e:
            self.logger.debug(f"Error generating regime recommendations: {e}")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _assess_regime_data_coverage(self, fund_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Assess data coverage for regime analysis."""
        
        coverage = {
            "funds_with_data": len(fund_data),
            "total_data_points": 0,
            "date_range": {},
            "data_quality": "good"
        }
        
        try:
            if fund_data:
                all_dates = []
                total_points = 0
                
                for data in fund_data.values():
                    if not data.empty:
                        all_dates.extend(data.index.tolist())
                        total_points += len(data)
                
                if all_dates:
                    all_dates = sorted(set(all_dates))
                    coverage["date_range"] = {
                        "start": all_dates[0].isoformat(),
                        "end": all_dates[-1].isoformat(),
                        "span_days": (all_dates[-1] - all_dates[0]).days
                    }
                    coverage["total_data_points"] = total_points
                    
                    # Assess quality based on data span
                    span_years = coverage["date_range"]["span_days"] / 365.25
                    if span_years >= 5:
                        coverage["data_quality"] = "excellent"
                    elif span_years >= 2:
                        coverage["data_quality"] = "good"
                    elif span_years >= 1:
                        coverage["data_quality"] = "fair"
                    else:
                        coverage["data_quality"] = "poor"
        
        except Exception as e:
            self.logger.debug(f"Error assessing data coverage: {e}")
            coverage["data_quality"] = "unknown"
        
        return coverage