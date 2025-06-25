"""Data quality assessment utilities."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DataQualityAssessment:
    """Comprehensive data quality assessment for financial data."""
    
    def __init__(self, min_quality_threshold: float = 0.7):
        """
        Initialize data quality assessment.
        
        Args:
            min_quality_threshold: Minimum quality score (0-1) for acceptable data
        """
        self.min_quality_threshold = min_quality_threshold
    
    def assess_data_quality(
        self,
        data: pd.DataFrame,
        price_column: str = "Close",
        return_column: str = "daily_return"
    ) -> float:
        """
        Comprehensive data quality assessment.
        
        Args:
            data: DataFrame with financial data
            price_column: Name of the price column
            return_column: Name of the return column
            
        Returns:
            Quality score between 0 and 1
        """
        if data.empty:
            return 0.0
        
        quality_scores = []
        
        # 1. Completeness check (25% weight)
        completeness_score = self._assess_completeness(data, price_column)
        quality_scores.append(("completeness", completeness_score, 0.25))
        
        # 2. Consistency check (20% weight)
        consistency_score = self._assess_consistency(data, price_column)
        quality_scores.append(("consistency", consistency_score, 0.20))
        
        # 3. Accuracy check (20% weight)
        accuracy_score = self._assess_accuracy(data, price_column, return_column)
        quality_scores.append(("accuracy", accuracy_score, 0.20))
        
        # 4. Timeliness check (15% weight)
        timeliness_score = self._assess_timeliness(data)
        quality_scores.append(("timeliness", timeliness_score, 0.15))
        
        # 5. Validity check (20% weight)
        validity_score = self._assess_validity(data, price_column, return_column)
        quality_scores.append(("validity", validity_score, 0.20))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in quality_scores)
        
        # Log detailed scores
        logger.debug(f"Data quality scores: {quality_scores}")
        logger.info(f"Overall data quality score: {total_score:.3f}")
        
        return min(1.0, max(0.0, total_score))
    
    def _assess_completeness(self, data: pd.DataFrame, price_column: str) -> float:
        """Assess data completeness."""
        
        if data.empty:
            return 0.0
        
        # Check for missing values in critical columns
        critical_columns = [price_column]
        if "Volume" in data.columns:
            critical_columns.append("Volume")
        
        total_missing = 0
        total_possible = 0
        
        for col in critical_columns:
            if col in data.columns:
                missing_count = data[col].isna().sum()
                total_missing += missing_count
                total_possible += len(data)
        
        if total_possible == 0:
            return 0.0
        
        completeness_ratio = 1 - (total_missing / total_possible)
        
        # Check for date gaps
        if len(data) > 1:
            date_gaps = self._detect_date_gaps(data.index)
            gap_penalty = min(0.3, len(date_gaps) * 0.05)  # Max 30% penalty
            completeness_ratio -= gap_penalty
        
        return max(0.0, completeness_ratio)
    
    def _assess_consistency(self, data: pd.DataFrame, price_column: str) -> float:
        """Assess data consistency."""
        
        if len(data) < 2:
            return 1.0
        
        consistency_score = 1.0
        
        # Check for price consistency (High >= Close >= Low)
        if all(col in data.columns for col in ["High", "Low", price_column]):
            inconsistent_ohlc = (
                (data["High"] < data[price_column]) |
                (data["Low"] > data[price_column]) |
                (data["High"] < data["Low"])
            ).sum()
            
            if len(data) > 0:
                ohlc_consistency = 1 - (inconsistent_ohlc / len(data))
                consistency_score *= ohlc_consistency
        
        # Check for unrealistic price jumps
        if price_column in data.columns:
            returns = data[price_column].pct_change().dropna()
            if len(returns) > 10:
                # Flag returns beyond 3 standard deviations
                outlier_threshold = 3
                outliers = np.abs(returns) > (returns.std() * outlier_threshold)
                outlier_ratio = outliers.sum() / len(returns)
                
                # Penalize if more than 1% outliers
                if outlier_ratio > 0.01:
                    consistency_score *= (1 - min(0.5, outlier_ratio * 10))
        
        return max(0.0, consistency_score)
    
    def _assess_accuracy(self, data: pd.DataFrame, price_column: str, return_column: str) -> float:
        """Assess data accuracy."""
        
        if len(data) < 2:
            return 1.0
        
        accuracy_score = 1.0
        
        # Check if calculated returns match provided returns
        if all(col in data.columns for col in [price_column, return_column]):
            calculated_returns = data[price_column].pct_change()
            provided_returns = data[return_column]
            
            # Find overlapping non-null values
            mask = calculated_returns.notna() & provided_returns.notna()
            if mask.sum() > 10:
                calc_vals = calculated_returns[mask]
                prov_vals = provided_returns[mask]
                
                # Calculate correlation and mean absolute error
                correlation = np.corrcoef(calc_vals, prov_vals)[0, 1]
                mae = np.mean(np.abs(calc_vals - prov_vals))
                
                # Good accuracy if correlation > 0.95 and MAE < 0.001
                if not np.isnan(correlation):
                    correlation_score = max(0, correlation)
                    mae_score = max(0, 1 - (mae * 1000))  # Scale MAE
                    accuracy_score = (correlation_score + mae_score) / 2
        
        return max(0.0, accuracy_score)
    
    def _assess_timeliness(self, data: pd.DataFrame) -> float:
        """Assess data timeliness."""
        
        if data.empty:
            return 0.0
        
        # Check how recent the latest data point is
        latest_date = data.index[-1]
        if isinstance(latest_date, str):
            latest_date = pd.to_datetime(latest_date)
        
        current_date = datetime.now()
        days_old = (current_date - latest_date).days
        
        # Full score if data is within 3 days
        if days_old <= 3:
            return 1.0
        # Linear decay to 0 over 30 days
        elif days_old <= 30:
            return 1.0 - (days_old - 3) / 27
        else:
            return 0.0
    
    def _assess_validity(self, data: pd.DataFrame, price_column: str, return_column: str) -> float:
        """Assess data validity."""
        
        if data.empty:
            return 0.0
        
        validity_score = 1.0
        
        # Check for negative prices
        if price_column in data.columns:
            negative_prices = (data[price_column] <= 0).sum()
            if len(data) > 0:
                validity_score *= (1 - negative_prices / len(data))
        
        # Check for extreme returns
        if return_column in data.columns:
            returns = data[return_column].dropna()
            if len(returns) > 0:
                # Flag returns beyond reasonable bounds (e.g., ±50% daily)
                extreme_returns = (np.abs(returns) > 0.5).sum()
                validity_score *= (1 - extreme_returns / len(returns))
        
        # Check for duplicate dates
        if len(data) > 1:
            duplicate_dates = data.index.duplicated().sum()
            validity_score *= (1 - duplicate_dates / len(data))
        
        return max(0.0, validity_score)
    
    def _detect_date_gaps(self, date_index: pd.DatetimeIndex) -> List[Tuple[datetime, datetime]]:
        """Detect gaps in date series."""
        
        if len(date_index) < 2:
            return []
        
        gaps = []
        sorted_dates = sorted(date_index)
        
        for i in range(1, len(sorted_dates)):
            prev_date = sorted_dates[i-1]
            curr_date = sorted_dates[i]
            
            # Check if gap is more than 7 days (accounting for weekends)
            if (curr_date - prev_date).days > 7:
                gaps.append((prev_date, curr_date))
        
        return gaps
    
    def generate_quality_report(
        self,
        data: pd.DataFrame,
        fund_name: str,
        price_column: str = "Close",
        return_column: str = "daily_return"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Args:
            data: DataFrame with financial data
            fund_name: Name of the fund
            price_column: Name of the price column
            return_column: Name of the return column
            
        Returns:
            Detailed quality report
        """
        
        overall_score = self.assess_data_quality(data, price_column, return_column)
        
        report = {
            "fund_name": fund_name,
            "overall_quality_score": overall_score,
            "quality_level": self._get_quality_level(overall_score),
            "assessment_date": datetime.now().isoformat(),
            "data_period": {
                "start_date": data.index[0].isoformat() if len(data) > 0 else None,
                "end_date": data.index[-1].isoformat() if len(data) > 0 else None,
                "total_records": len(data)
            },
            "detailed_scores": {
                "completeness": self._assess_completeness(data, price_column),
                "consistency": self._assess_consistency(data, price_column),
                "accuracy": self._assess_accuracy(data, price_column, return_column),
                "timeliness": self._assess_timeliness(data),
                "validity": self._assess_validity(data, price_column, return_column)
            },
            "issues_found": self._identify_issues(data, price_column, return_column),
            "recommendations": self._generate_recommendations(data, overall_score)
        }
        
        return report
    
    def _get_quality_level(self, score: float) -> str:
        """Get quality level description."""
        
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "good"
        elif score >= 0.7:
            return "acceptable"
        elif score >= 0.5:
            return "poor"
        else:
            return "unacceptable"
    
    def _identify_issues(
        self,
        data: pd.DataFrame,
        price_column: str,
        return_column: str
    ) -> List[Dict[str, Any]]:
        """Identify specific data quality issues."""
        
        issues = []
        
        if data.empty:
            issues.append({
                "type": "critical",
                "category": "completeness",
                "description": "No data available"
            })
            return issues
        
        # Missing data issues
        if price_column in data.columns:
            missing_prices = data[price_column].isna().sum()
            if missing_prices > 0:
                issues.append({
                    "type": "warning",
                    "category": "completeness",
                    "description": f"{missing_prices} missing price values",
                    "count": missing_prices
                })
        
        # Date gap issues
        date_gaps = self._detect_date_gaps(data.index)
        if date_gaps:
            issues.append({
                "type": "warning",
                "category": "completeness",
                "description": f"{len(date_gaps)} significant date gaps found",
                "gaps": [(gap[0].isoformat(), gap[1].isoformat()) for gap in date_gaps[:5]]
            })
        
        # Outlier issues
        if return_column in data.columns:
            returns = data[return_column].dropna()
            if len(returns) > 10:
                outliers = np.abs(returns) > (returns.std() * 3)
                outlier_count = outliers.sum()
                if outlier_count > len(returns) * 0.01:  # More than 1%
                    issues.append({
                        "type": "warning",
                        "category": "consistency",
                        "description": f"{outlier_count} potential outliers in returns",
                        "outlier_percentage": (outlier_count / len(returns)) * 100
                    })
        
        # Timeliness issues
        if len(data) > 0:
            latest_date = data.index[-1]
            days_old = (datetime.now() - latest_date).days
            if days_old > 7:
                issues.append({
                    "type": "warning" if days_old <= 30 else "critical",
                    "category": "timeliness",
                    "description": f"Data is {days_old} days old",
                    "days_old": days_old
                })
        
        return issues
    
    def _generate_recommendations(self, data: pd.DataFrame, quality_score: float) -> List[str]:
        """Generate data quality improvement recommendations."""
        
        recommendations = []
        
        if quality_score < self.min_quality_threshold:
            recommendations.append(
                f"Overall quality score ({quality_score:.2f}) is below threshold "
                f"({self.min_quality_threshold}). Consider alternative data sources."
            )
        
        if data.empty:
            recommendations.append("No data available. Check data source connectivity.")
            return recommendations
        
        # Completeness recommendations
        completeness_score = self._assess_completeness(data, "Close")
        if completeness_score < 0.9:
            recommendations.append(
                "Consider data interpolation or alternative sources for missing values."
            )
        
        # Timeliness recommendations
        timeliness_score = self._assess_timeliness(data)
        if timeliness_score < 0.8:
            recommendations.append(
                "Update data collection frequency to ensure timely data."
            )
        
        # Consistency recommendations
        consistency_score = self._assess_consistency(data, "Close")
        if consistency_score < 0.9:
            recommendations.append(
                "Implement data validation rules to catch inconsistencies."
            )
        
        if quality_score >= 0.9:
            recommendations.append("Data quality is excellent. No immediate action needed.")
        
        return recommendations
    
    def compare_data_sources(
        self,
        data_sources: Dict[str, pd.DataFrame],
        fund_name: str
    ) -> Dict[str, Any]:
        """Compare quality across multiple data sources."""
        
        comparison = {
            "fund_name": fund_name,
            "sources_compared": list(data_sources.keys()),
            "comparison_date": datetime.now().isoformat(),
            "source_scores": {},
            "best_source": None,
            "recommendation": None
        }
        
        best_score = 0
        best_source = None
        
        for source_name, data in data_sources.items():
            score = self.assess_data_quality(data)
            comparison["source_scores"][source_name] = {
                "quality_score": score,
                "record_count": len(data),
                "date_range": {
                    "start": data.index[0].isoformat() if len(data) > 0 else None,
                    "end": data.index[-1].isoformat() if len(data) > 0 else None
                }
            }
            
            if score > best_score:
                best_score = score
                best_source = source_name
        
        comparison["best_source"] = best_source
        
        # Generate recommendation
        if best_score >= 0.9:
            comparison["recommendation"] = f"Use {best_source} as primary source (excellent quality)"
        elif best_score >= 0.7:
            comparison["recommendation"] = f"Use {best_source} as primary source with monitoring"
        else:
            comparison["recommendation"] = "Consider alternative data sources - all current sources have quality issues"
        
        return comparison