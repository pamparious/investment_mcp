"""
Base collector class with shared functionality for all data collectors.

This module provides common functionality and patterns used across all
data collection modules to ensure consistency and reduce code duplication.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from sqlalchemy.orm import Session
from ..core.config import get_settings

logger = logging.getLogger(__name__)


class BaseDataCollector(ABC):
    """Abstract base class for all data collectors."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    async def collect_historical_data(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """Collect historical data. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def collect_recent_data(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """Collect recent data for updates. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def save_to_database(self, data_dict: Dict[str, pd.DataFrame], db_session: Session) -> int:
        """Save collected data to database. Must be implemented by subclasses."""
        pass
    
    def validate_data_quality(
        self, 
        data: pd.DataFrame, 
        required_columns: List[str],
        min_records: int = 10
    ) -> Dict[str, Any]:
        """
        Validate data quality and return quality metrics.
        
        Args:
            data: DataFrame to validate
            required_columns: List of columns that must be present
            min_records: Minimum number of records required
            
        Returns:
            Dictionary with validation results and quality metrics
        """
        
        if data.empty:
            return {
                "valid": False,
                "errors": ["Data is empty"],
                "warnings": [],
                "metrics": {"record_count": 0, "completeness": 0.0}
            }
        
        errors = []
        warnings = []
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check minimum record count
        record_count = len(data)
        if record_count < min_records:
            warnings.append(f"Low record count: {record_count} (minimum: {min_records})")
        
        # Calculate completeness
        total_cells = len(data) * len(data.columns)
        missing_cells = data.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0.0
        
        # Check for completeness threshold
        threshold = self.settings.DATA_QUALITY_THRESHOLD
        if completeness < threshold:
            warnings.append(f"Data completeness {completeness:.1%} below threshold {threshold:.1%}")
        
        # Date column validation
        if 'date' in data.columns:
            date_col = data['date']
            if date_col.isnull().any():
                errors.append("Date column contains null values")
            
            # Check for future dates
            future_dates = date_col > datetime.now().date()
            if future_dates.any():
                warnings.append(f"Data contains {future_dates.sum()} future dates")
        
        # Numeric column validation
        numeric_columns = data.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            if col in data.columns:
                # Check for infinite values
                if pd.isinf(data[col]).any():
                    errors.append(f"Column {col} contains infinite values")
                
                # Check for negative values where inappropriate
                if col in ['volume', 'price', 'close', 'open', 'high', 'low']:
                    if (data[col] < 0).any():
                        warnings.append(f"Column {col} contains negative values")
        
        is_valid = len(errors) == 0 and completeness >= threshold
        
        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "metrics": {
                "record_count": record_count,
                "completeness": completeness,
                "missing_cells": int(missing_cells),
                "total_cells": int(total_cells)
            }
        }
    
    def standardize_date_column(self, data: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """Standardize date column format across all collectors."""
        
        if date_column not in data.columns:
            self.logger.warning(f"Date column '{date_column}' not found in data")
            return data
        
        try:
            # Convert to datetime if not already
            data[date_column] = pd.to_datetime(data[date_column])
            
            # Convert to date objects for consistency
            data[date_column] = data[date_column].dt.date
            
            # Sort by date
            data = data.sort_values(date_column)
            
            self.logger.debug(f"Standardized {date_column} column for {len(data)} records")
            
        except Exception as e:
            self.logger.error(f"Error standardizing date column: {e}")
        
        return data
    
    def remove_duplicates(
        self, 
        data: pd.DataFrame, 
        subset: Optional[List[str]] = None,
        keep: str = 'last'
    ) -> pd.DataFrame:
        """Remove duplicate records, keeping the most recent."""
        
        if data.empty:
            return data
        
        initial_count = len(data)
        
        # Default subset for duplicate detection
        if subset is None:
            if 'date' in data.columns and 'symbol' in data.columns:
                subset = ['date', 'symbol']
            elif 'date' in data.columns:
                subset = ['date']
        
        if subset:
            data = data.drop_duplicates(subset=subset, keep=keep)
            
            removed_count = initial_count - len(data)
            if removed_count > 0:
                self.logger.info(f"Removed {removed_count} duplicate records")
        
        return data
    
    def handle_missing_values(
        self, 
        data: pd.DataFrame, 
        method: str = 'forward_fill',
        numeric_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Handle missing values using specified method."""
        
        if data.empty:
            return data
        
        if numeric_columns is None:
            numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
        
        initial_missing = data.isnull().sum().sum()
        
        if method == 'forward_fill':
            # Forward fill for time series data
            data[numeric_columns] = data[numeric_columns].fillna(method='ffill')
            
        elif method == 'interpolate':
            # Linear interpolation for numeric columns
            data[numeric_columns] = data[numeric_columns].interpolate(method='linear')
            
        elif method == 'drop':
            # Drop rows with any missing values
            data = data.dropna()
            
        elif method == 'zero_fill':
            # Fill with zeros (useful for volume data)
            data[numeric_columns] = data[numeric_columns].fillna(0)
        
        final_missing = data.isnull().sum().sum()
        handled_missing = initial_missing - final_missing
        
        if handled_missing > 0:
            self.logger.info(f"Handled {handled_missing} missing values using {method}")
        
        return data
    
    def create_collection_summary(
        self, 
        data_dict: Dict[str, pd.DataFrame],
        collection_type: str
    ) -> Dict[str, Any]:
        """Create summary of data collection results."""
        
        summary = {
            "collection_type": collection_type,
            "timestamp": datetime.utcnow().isoformat(),
            "total_datasets": len(data_dict),
            "successful_collections": 0,
            "failed_collections": 0,
            "total_records": 0,
            "datasets": {}
        }
        
        for dataset_name, data in data_dict.items():
            if data.empty:
                summary["failed_collections"] += 1
                summary["datasets"][dataset_name] = {
                    "status": "failed",
                    "records": 0,
                    "date_range": None
                }
            else:
                summary["successful_collections"] += 1
                summary["total_records"] += len(data)
                
                date_range = None
                if 'date' in data.columns:
                    date_range = {
                        "start": str(data['date'].min()),
                        "end": str(data['date'].max())
                    }
                
                summary["datasets"][dataset_name] = {
                    "status": "success",
                    "records": len(data),
                    "date_range": date_range,
                    "columns": list(data.columns)
                }
        
        return summary
    
    async def retry_with_backoff(
        self,
        coro_func,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0
    ):
        """Retry an async function with exponential backoff."""
        
        last_exception = None
        delay = initial_delay
        
        for attempt in range(max_retries + 1):
            try:
                return await coro_func()
            except Exception as e:
                last_exception = e
                
                if attempt == max_retries:
                    self.logger.error(f"Failed after {max_retries + 1} attempts: {e}")
                    raise e
                
                self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                await asyncio.sleep(delay)
                delay *= backoff_factor
        
        # This should never be reached, but just in case
        raise last_exception
    
    def log_collection_metrics(
        self,
        collection_type: str,
        start_time: datetime,
        end_time: datetime,
        records_collected: int,
        errors: List[str] = None
    ):
        """Log standardized collection metrics."""
        
        duration = (end_time - start_time).total_seconds()
        
        self.logger.info(
            f"Collection completed - Type: {collection_type}, "
            f"Duration: {duration:.1f}s, "
            f"Records: {records_collected}, "
            f"Rate: {records_collected/duration:.1f} records/sec"
        )
        
        if errors:
            self.logger.warning(f"Collection had {len(errors)} errors: {errors[:3]}")  # Log first 3 errors