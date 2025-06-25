"""Fundamental analysis engine for Swedish economic data correlation."""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from ...utils.exceptions import AnalysisError

logger = logging.getLogger(__name__)


class FundamentalAnalyzer:
    """Fundamental analysis engine integrating economic indicators with fund performance."""
    
    def __init__(self):
        """Initialize the fundamental analyzer."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.scaler = StandardScaler()
    
    def analyze_economic_correlations(
        self,
        fund_data: Dict[str, pd.DataFrame],
        economic_data: Dict[str, Any],
        correlation_window: int = 252
    ) -> Dict[str, Any]:
        """
        Analyze correlations between fund performance and economic indicators.
        
        Args:
            fund_data: Dictionary of fund DataFrames
            economic_data: Economic data from Riksbank/SCB
            correlation_window: Rolling window for correlation calculation
            
        Returns:
            Dictionary with correlation analysis results
        """
        try:
            self.logger.info("Starting economic correlation analysis")
            
            correlations = {}
            
            for fund_name, data in fund_data.items():
                if data.empty or 'daily_return' not in data.columns:
                    continue
                
                fund_correlations = self._calculate_fund_correlations(
                    data, economic_data, correlation_window
                )
                correlations[fund_name] = fund_correlations
            
            # Calculate aggregate correlations across all funds
            aggregate_correlations = self._calculate_aggregate_correlations(correlations)
            
            # Identify key economic drivers
            key_drivers = self._identify_key_drivers(correlations)
            
            # Calculate regime-dependent correlations
            regime_correlations = self._analyze_regime_correlations(fund_data, economic_data)
            
            results = {
                "individual_fund_correlations": correlations,
                "aggregate_correlations": aggregate_correlations,
                "key_economic_drivers": key_drivers,
                "regime_dependent_correlations": regime_correlations,
                "analysis_metadata": {
                    "correlation_window": correlation_window,
                    "funds_analyzed": len(correlations),
                    "economic_indicators": list(self._extract_economic_indicators(economic_data).keys())
                }
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in economic correlation analysis: {e}")
            raise AnalysisError(f"Economic correlation analysis failed: {e}")
    
    def _calculate_fund_correlations(
        self,
        fund_data: pd.DataFrame,
        economic_data: Dict[str, Any],
        window: int
    ) -> Dict[str, Any]:
        """Calculate correlations for a single fund."""
        
        # Extract economic indicators
        economic_indicators = self._extract_economic_indicators(economic_data)
        
        if not economic_indicators:
            return {"error": "No economic indicators available"}
        
        # Prepare fund returns
        fund_returns = fund_data['daily_return'].dropna()
        
        correlations = {}
        
        for indicator_name, indicator_data in economic_indicators.items():
            try:
                correlation_result = self._calculate_indicator_correlation(
                    fund_returns, indicator_data, window
                )
                correlations[indicator_name] = correlation_result
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate correlation with {indicator_name}: {e}")
                correlations[indicator_name] = {"error": str(e)}
        
        return correlations
    
    def _extract_economic_indicators(self, economic_data: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Extract time series data from economic data structure."""
        
        indicators = {}
        
        try:
            # Extract Riksbank data
            if "repo_rate" in economic_data:
                repo_data = economic_data["repo_rate"].get("data", [])
                if repo_data:
                    df = pd.DataFrame(repo_data)
                    if "date" in df.columns and "value" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                        df = df.set_index("date").sort_index()
                        indicators["repo_rate"] = df["value"]
            
            # Extract government bond yields
            if "government_bonds" in economic_data:
                for maturity, bond_data in economic_data["government_bonds"].items():
                    bond_series = bond_data.get("data", [])
                    if bond_series:
                        df = pd.DataFrame(bond_series)
                        if "date" in df.columns and "value" in df.columns:
                            df["date"] = pd.to_datetime(df["date"])
                            df = df.set_index("date").sort_index()
                            indicators[f"gov_bond_{maturity}"] = df["value"]
            
            # Extract exchange rates
            if "exchange_rates" in economic_data:
                for currency, rate_data in economic_data["exchange_rates"].items():
                    rate_series = rate_data.get("data", [])
                    if rate_series:
                        df = pd.DataFrame(rate_series)
                        if "date" in df.columns and "value" in df.columns:
                            df["date"] = pd.to_datetime(df["date"])
                            df = df.set_index("date").sort_index()
                            indicators[currency.lower()] = df["value"]
            
            # Calculate derived indicators
            if "repo_rate" in indicators and "gov_bond_10y" in indicators:
                # Term spread
                indicators["term_spread"] = indicators["gov_bond_10y"] - indicators["repo_rate"]
            
            if "sek_usd" in indicators:
                # USD strength (inverse of SEK/USD)
                indicators["usd_strength"] = 1 / indicators["sek_usd"]
            
        except Exception as e:
            self.logger.error(f"Error extracting economic indicators: {e}")
        
        return indicators
    
    def _calculate_indicator_correlation(
        self,
        fund_returns: pd.Series,
        indicator_data: pd.Series,
        window: int
    ) -> Dict[str, Any]:
        """Calculate correlation between fund returns and economic indicator."""
        
        # Align data by date
        aligned_data = pd.concat([fund_returns, indicator_data], axis=1, join='inner')
        aligned_data.columns = ['returns', 'indicator']
        aligned_data = aligned_data.dropna()
        
        if len(aligned_data) < window:
            return {
                "static_correlation": None,
                "rolling_correlation": None,
                "significance": None,
                "data_points": len(aligned_data)
            }
        
        returns = aligned_data['returns']
        indicator = aligned_data['indicator']
        
        # Calculate static correlation
        static_corr, p_value = stats.pearsonr(returns, indicator)
        
        # Calculate rolling correlation
        rolling_corr = aligned_data['returns'].rolling(window).corr(aligned_data['indicator'])
        
        # Calculate correlation statistics
        correlation_stats = {
            "mean": float(rolling_corr.mean()) if not rolling_corr.empty else None,
            "std": float(rolling_corr.std()) if not rolling_corr.empty else None,
            "min": float(rolling_corr.min()) if not rolling_corr.empty else None,
            "max": float(rolling_corr.max()) if not rolling_corr.empty else None,
            "current": float(rolling_corr.iloc[-1]) if not rolling_corr.empty else None
        }
        
        # Determine correlation strength
        abs_corr = abs(static_corr)
        if abs_corr > 0.7:
            strength = "strong"
        elif abs_corr > 0.3:
            strength = "moderate"
        elif abs_corr > 0.1:
            strength = "weak"
        else:
            strength = "negligible"
        
        return {
            "static_correlation": float(static_corr),
            "rolling_correlation_stats": correlation_stats,
            "significance": float(p_value),
            "correlation_strength": strength,
            "data_points": len(aligned_data),
            "is_significant": p_value < 0.05
        }
    
    def _calculate_aggregate_correlations(self, fund_correlations: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate correlations across all funds."""
        
        if not fund_correlations:
            return {}
        
        # Collect all economic indicators
        all_indicators = set()
        for fund_data in fund_correlations.values():
            if isinstance(fund_data, dict):
                all_indicators.update(fund_data.keys())
        
        aggregate_correlations = {}
        
        for indicator in all_indicators:
            correlations = []
            significance_scores = []
            
            for fund_name, fund_data in fund_correlations.items():
                if indicator in fund_data and isinstance(fund_data[indicator], dict):
                    corr_data = fund_data[indicator]
                    static_corr = corr_data.get("static_correlation")
                    p_value = corr_data.get("significance")
                    
                    if static_corr is not None and not np.isnan(static_corr):
                        correlations.append(static_corr)
                        if p_value is not None:
                            significance_scores.append(1 - p_value)  # Higher for more significant
            
            if correlations:
                aggregate_correlations[indicator] = {
                    "mean_correlation": float(np.mean(correlations)),
                    "median_correlation": float(np.median(correlations)),
                    "std_correlation": float(np.std(correlations)),
                    "min_correlation": float(np.min(correlations)),
                    "max_correlation": float(np.max(correlations)),
                    "funds_with_data": len(correlations),
                    "average_significance": float(np.mean(significance_scores)) if significance_scores else None
                }
        
        return aggregate_correlations
    
    def _identify_key_drivers(self, fund_correlations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify key economic drivers based on correlation strength and consistency."""
        
        aggregate_correlations = self._calculate_aggregate_correlations(fund_correlations)
        
        # Score each indicator
        driver_scores = []
        
        for indicator, stats in aggregate_correlations.items():
            if isinstance(stats, dict):
                mean_abs_corr = abs(stats.get("mean_correlation", 0))
                consistency = 1 - stats.get("std_correlation", 1)  # Lower std = higher consistency
                significance = stats.get("average_significance", 0) or 0
                coverage = stats.get("funds_with_data", 0) / len(fund_correlations)
                
                # Composite score
                score = (mean_abs_corr * 0.4 + consistency * 0.3 + significance * 0.2 + coverage * 0.1)
                
                driver_scores.append({
                    "indicator": indicator,
                    "score": score,
                    "mean_correlation": stats.get("mean_correlation"),
                    "consistency": consistency,
                    "significance": significance,
                    "fund_coverage": coverage
                })
        
        # Sort by score and return top drivers
        driver_scores.sort(key=lambda x: x["score"], reverse=True)
        
        return driver_scores[:10]  # Top 10 drivers
    
    def _analyze_regime_correlations(
        self,
        fund_data: Dict[str, pd.DataFrame],
        economic_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze correlations in different market regimes."""
        
        try:
            # Define market regimes based on volatility and trends
            regime_data = {}
            
            for fund_name, data in fund_data.items():
                if data.empty or 'daily_return' not in data.columns:
                    continue
                
                # Calculate regime indicators
                volatility = data['daily_return'].rolling(30).std()
                trend = data['Close'].pct_change(60)
                
                # Define regimes
                high_vol = volatility > volatility.quantile(0.75)
                low_vol = volatility < volatility.quantile(0.25)
                bull_market = trend > 0.1
                bear_market = trend < -0.1
                
                regimes = pd.Series(index=data.index, dtype=str)
                regimes.loc[high_vol & bull_market] = "high_vol_bull"
                regimes.loc[high_vol & bear_market] = "high_vol_bear"
                regimes.loc[low_vol & bull_market] = "low_vol_bull"
                regimes.loc[low_vol & bear_market] = "low_vol_bear"
                regimes.fillna("neutral", inplace=True)
                
                # Calculate correlations by regime
                economic_indicators = self._extract_economic_indicators(economic_data)
                
                fund_regime_correlations = {}
                for regime_type in regimes.unique():
                    regime_mask = regimes == regime_type
                    regime_returns = data.loc[regime_mask, 'daily_return']
                    
                    regime_correlations = {}
                    for indicator_name, indicator_data in economic_indicators.items():
                        try:
                            aligned_regime_data = pd.concat([regime_returns, indicator_data], axis=1, join='inner')
                            aligned_regime_data.columns = ['returns', 'indicator']
                            aligned_regime_data = aligned_regime_data.dropna()
                            
                            if len(aligned_regime_data) > 10:  # Minimum data points
                                corr, p_val = stats.pearsonr(
                                    aligned_regime_data['returns'],
                                    aligned_regime_data['indicator']
                                )
                                regime_correlations[indicator_name] = {
                                    "correlation": float(corr),
                                    "significance": float(p_val),
                                    "data_points": len(aligned_regime_data)
                                }
                        except Exception as e:
                            self.logger.debug(f"Regime correlation calculation failed for {indicator_name}: {e}")
                    
                    fund_regime_correlations[regime_type] = regime_correlations
                
                regime_data[fund_name] = fund_regime_correlations
            
            return regime_data
            
        except Exception as e:
            self.logger.error(f"Error in regime correlation analysis: {e}")
            return {"error": str(e)}
    
    def analyze_sector_fundamentals(
        self,
        fund_data: Dict[str, pd.DataFrame],
        economic_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze sector-specific fundamental factors."""
        
        try:
            # Group funds by category/sector
            fund_categories = self._categorize_funds(fund_data.keys())
            
            sector_analysis = {}
            
            for category, fund_names in fund_categories.items():
                category_data = {name: fund_data[name] for name in fund_names if name in fund_data}
                
                if not category_data:
                    continue
                
                # Calculate sector-level metrics
                sector_metrics = self._calculate_sector_metrics(category_data, economic_data)
                sector_analysis[category] = sector_metrics
            
            return sector_analysis
            
        except Exception as e:
            self.logger.error(f"Error in sector fundamental analysis: {e}")
            raise AnalysisError(f"Sector fundamental analysis failed: {e}")
    
    def _categorize_funds(self, fund_names: List[str]) -> Dict[str, List[str]]:
        """Categorize funds by type/sector."""
        
        categories = {
            "equity": [],
            "global": [],
            "emerging": [],
            "technology": [],
            "small_cap": [],
            "sustainable": []
        }
        
        for fund_name in fund_names:
            fund_lower = fund_name.lower()
            
            if any(keyword in fund_lower for keyword in ["global", "världen", "world"]):
                categories["global"].append(fund_name)
            elif any(keyword in fund_lower for keyword in ["emerging", "tillväxt"]):
                categories["emerging"].append(fund_name)
            elif any(keyword in fund_lower for keyword in ["tech", "teknologi"]):
                categories["technology"].append(fund_name)
            elif any(keyword in fund_lower for keyword in ["småbolag", "small"]):
                categories["small_cap"].append(fund_name)
            elif any(keyword in fund_lower for keyword in ["hållbar", "sustainable", "esg"]):
                categories["sustainable"].append(fund_name)
            else:
                categories["equity"].append(fund_name)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def _calculate_sector_metrics(
        self,
        sector_funds: Dict[str, pd.DataFrame],
        economic_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate metrics for a fund sector/category."""
        
        if not sector_funds:
            return {}
        
        # Calculate sector returns (equal-weighted)
        all_returns = []
        for fund_data in sector_funds.values():
            if 'daily_return' in fund_data.columns:
                returns = fund_data['daily_return'].dropna()
                all_returns.append(returns)
        
        if not all_returns:
            return {"error": "No return data available"}
        
        # Align all return series
        aligned_returns = pd.concat(all_returns, axis=1, join='inner')
        sector_returns = aligned_returns.mean(axis=1)
        
        # Calculate sector-level correlations with economic indicators
        economic_indicators = self._extract_economic_indicators(economic_data)
        
        sector_correlations = {}
        for indicator_name, indicator_data in economic_indicators.items():
            try:
                correlation_result = self._calculate_indicator_correlation(
                    sector_returns, indicator_data, window=60
                )
                sector_correlations[indicator_name] = correlation_result
            except Exception as e:
                self.logger.debug(f"Sector correlation calculation failed for {indicator_name}: {e}")
        
        # Calculate sector performance metrics
        sector_metrics = {
            "fund_count": len(sector_funds),
            "average_correlation": sector_correlations,
            "sector_volatility": float(sector_returns.std() * np.sqrt(252)),
            "sector_sharpe": self._calculate_sharpe_ratio(sector_returns),
            "sector_max_drawdown": self._calculate_max_drawdown(sector_returns),
            "economic_sensitivity": self._assess_economic_sensitivity(sector_correlations)
        }
        
        return sector_metrics
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for a return series."""
        
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)
        return float((excess_returns.mean() * 252) / (returns.std() * np.sqrt(252)))
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown for a return series."""
        
        if len(returns) < 2:
            return 0.0
        
        cumulative = (1 + returns.fillna(0)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min())
    
    def _assess_economic_sensitivity(self, correlations: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall economic sensitivity based on correlations."""
        
        significant_correlations = []
        
        for indicator, corr_data in correlations.items():
            if isinstance(corr_data, dict):
                corr_value = corr_data.get("static_correlation")
                is_significant = corr_data.get("is_significant", False)
                
                if corr_value is not None and is_significant:
                    significant_correlations.append(abs(corr_value))
        
        if not significant_correlations:
            return {
                "sensitivity_level": "low",
                "average_sensitivity": 0.0,
                "significant_factors": 0
            }
        
        avg_sensitivity = np.mean(significant_correlations)
        
        if avg_sensitivity > 0.5:
            sensitivity_level = "high"
        elif avg_sensitivity > 0.3:
            sensitivity_level = "medium"
        else:
            sensitivity_level = "low"
        
        return {
            "sensitivity_level": sensitivity_level,
            "average_sensitivity": float(avg_sensitivity),
            "significant_factors": len(significant_correlations)
        }
    
    def generate_fundamental_report(
        self,
        fund_data: Dict[str, pd.DataFrame],
        economic_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive fundamental analysis report."""
        
        try:
            self.logger.info("Generating comprehensive fundamental analysis report")
            
            # Economic correlations
            economic_correlations = self.analyze_economic_correlations(fund_data, economic_data)
            
            # Sector analysis
            sector_analysis = self.analyze_sector_fundamentals(fund_data, economic_data)
            
            # Market regime analysis
            regime_analysis = economic_correlations.get("regime_dependent_correlations", {})
            
            # Key insights
            key_insights = self._generate_key_insights(economic_correlations, sector_analysis)
            
            report = {
                "executive_summary": {
                    "total_funds_analyzed": len(fund_data),
                    "key_economic_drivers": economic_correlations.get("key_economic_drivers", [])[:3],
                    "highest_sensitivity_sector": self._find_highest_sensitivity_sector(sector_analysis),
                    "market_regime_insights": self._summarize_regime_insights(regime_analysis)
                },
                "detailed_analysis": {
                    "economic_correlations": economic_correlations,
                    "sector_fundamentals": sector_analysis,
                    "regime_analysis": regime_analysis
                },
                "key_insights": key_insights,
                "recommendations": self._generate_recommendations(economic_correlations, sector_analysis),
                "report_metadata": {
                    "analysis_date": pd.Timestamp.now().isoformat(),
                    "data_coverage": self._assess_data_coverage(fund_data, economic_data)
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating fundamental report: {e}")
            raise AnalysisError(f"Fundamental report generation failed: {e}")
    
    def _generate_key_insights(
        self,
        economic_correlations: Dict[str, Any],
        sector_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate key insights from the analysis."""
        
        insights = []
        
        # Economic driver insights
        key_drivers = economic_correlations.get("key_economic_drivers", [])
        if key_drivers:
            top_driver = key_drivers[0]
            insights.append(
                f"The strongest economic driver is {top_driver['indicator']} "
                f"with an average correlation of {top_driver.get('mean_correlation', 0):.2f}"
            )
        
        # Sector insights
        if sector_analysis:
            most_sensitive_sector = max(
                sector_analysis.items(),
                key=lambda x: x[1].get("economic_sensitivity", {}).get("average_sensitivity", 0)
            )
            insights.append(
                f"The {most_sensitive_sector[0]} sector shows the highest economic sensitivity "
                f"({most_sensitive_sector[1].get('economic_sensitivity', {}).get('sensitivity_level', 'unknown')} level)"
            )
        
        return insights
    
    def _find_highest_sensitivity_sector(self, sector_analysis: Dict[str, Any]) -> str:
        """Find the sector with highest economic sensitivity."""
        
        if not sector_analysis:
            return "unknown"
        
        highest_sensitivity = 0
        highest_sector = "unknown"
        
        for sector, data in sector_analysis.items():
            if isinstance(data, dict):
                sensitivity = data.get("economic_sensitivity", {}).get("average_sensitivity", 0)
                if sensitivity > highest_sensitivity:
                    highest_sensitivity = sensitivity
                    highest_sector = sector
        
        return highest_sector
    
    def _summarize_regime_insights(self, regime_analysis: Dict[str, Any]) -> str:
        """Summarize market regime insights."""
        
        if not regime_analysis:
            return "No regime analysis available"
        
        # Count regime types across all funds
        regime_counts = {}
        for fund_data in regime_analysis.values():
            if isinstance(fund_data, dict):
                for regime in fund_data.keys():
                    regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        if regime_counts:
            most_common_regime = max(regime_counts.items(), key=lambda x: x[1])
            return f"Most funds show distinct behavior in {most_common_regime[0]} markets"
        
        return "Regime patterns vary significantly across funds"
    
    def _generate_recommendations(
        self,
        economic_correlations: Dict[str, Any],
        sector_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate investment recommendations based on fundamental analysis."""
        
        recommendations = []
        
        # Economic sensitivity recommendations
        key_drivers = economic_correlations.get("key_economic_drivers", [])
        if key_drivers:
            top_driver = key_drivers[0]
            recommendations.append(
                f"Monitor {top_driver['indicator']} closely as it shows the strongest "
                f"correlation with fund performance"
            )
        
        # Sector diversification recommendations
        if sector_analysis:
            sensitivities = []
            for sector, data in sector_analysis.items():
                if isinstance(data, dict):
                    sensitivity = data.get("economic_sensitivity", {}).get("average_sensitivity", 0)
                    sensitivities.append((sector, sensitivity))
            
            if sensitivities:
                sensitivities.sort(key=lambda x: x[1])
                least_sensitive = sensitivities[0]
                most_sensitive = sensitivities[-1]
                
                recommendations.append(
                    f"Consider {least_sensitive[0]} funds for defensive positioning "
                    f"and {most_sensitive[0]} funds for economic momentum plays"
                )
        
        return recommendations
    
    def _assess_data_coverage(
        self,
        fund_data: Dict[str, pd.DataFrame],
        economic_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess data coverage quality."""
        
        coverage = {
            "funds_with_data": len(fund_data),
            "economic_indicators_available": len(self._extract_economic_indicators(economic_data)),
            "average_data_length": 0,
            "data_quality": "good"
        }
        
        if fund_data:
            lengths = [len(df) for df in fund_data.values() if not df.empty]
            if lengths:
                coverage["average_data_length"] = int(np.mean(lengths))
        
        return coverage