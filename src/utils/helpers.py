"""
Utility helpers for Investment MCP System.

This module provides common utility functions used across
the Investment MCP system components.
"""

import json
import asyncio
import hashlib
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Optional, Union, Callable, TypeVar, Awaitable
from pathlib import Path
import pandas as pd
import numpy as np

T = TypeVar('T')


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float with default fallback."""
    try:
        if value is None or value == '':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int with default fallback."""
    try:
        if value is None or value == '':
            return default
        return int(float(value))  # Handle string floats like "123.0"
    except (ValueError, TypeError):
        return default


def safe_percentage(value: Any, default: float = 0.0) -> float:
    """Safely convert value to percentage (0-1 range)."""
    try:
        if value is None or value == '':
            return default
        
        float_val = float(value)
        
        # If value is > 1, assume it's already in percentage and convert
        if float_val > 1:
            float_val = float_val / 100
        
        # Clamp to valid percentage range
        return max(0.0, min(1.0, float_val))
    except (ValueError, TypeError):
        return default


def format_currency(amount: float, currency: str = "SEK", decimals: int = 2) -> str:
    """Format amount as currency string."""
    if currency == "SEK":
        if amount >= 1_000_000:
            return f"{amount / 1_000_000:.1f}M {currency}"
        elif amount >= 1_000:
            return f"{amount / 1_000:.1f}K {currency}"
        else:
            return f"{amount:,.{decimals}f} {currency}"
    else:
        return f"{amount:,.{decimals}f} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format decimal as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_date(date_obj: Union[datetime, date, str], format_str: str = "%Y-%m-%d") -> str:
    """Format date object as string."""
    if isinstance(date_obj, str):
        return date_obj
    elif isinstance(date_obj, datetime):
        return date_obj.strftime(format_str)
    elif isinstance(date_obj, date):
        return date_obj.strftime(format_str)
    else:
        return str(date_obj)


def parse_date(date_str: str, formats: List[str] = None) -> Optional[datetime]:
    """Parse date string with multiple format attempts."""
    if formats is None:
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d", 
            "%d/%m/%Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f"
        ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None


def calculate_hash(data: Any) -> str:
    """Calculate MD5 hash of data for caching/comparison."""
    if isinstance(data, dict):
        data_str = json.dumps(data, sort_keys=True)
    elif isinstance(data, (list, tuple)):
        data_str = json.dumps(list(data), sort_keys=True)
    else:
        data_str = str(data)
    
    return hashlib.md5(data_str.encode()).hexdigest()


def chunk_list(data: List[T], chunk_size: int) -> List[List[T]]:
    """Split list into chunks of specified size."""
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """Unflatten dictionary with dotted keys."""
    result = {}
    for key, value in d.items():
        keys = key.split(sep)
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    return result


def remove_none_values(d: Dict[str, Any], recursive: bool = True) -> Dict[str, Any]:
    """Remove None values from dictionary."""
    result = {}
    for k, v in d.items():
        if v is not None:
            if recursive and isinstance(v, dict):
                result[k] = remove_none_values(v, recursive)
            else:
                result[k] = v
    return result


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries with later ones taking precedence."""
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


async def retry_async(
    func: Callable[..., Awaitable[T]],
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> T:
    """Retry async function with exponential backoff."""
    
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_attempts):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            
            if attempt == max_attempts - 1:
                raise e
            
            await asyncio.sleep(current_delay)
            current_delay *= backoff_factor
    
    # This should never be reached, but just in case
    raise last_exception


def retry_sync(
    func: Callable[..., T],
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> T:
    """Retry synchronous function with exponential backoff."""
    
    import time
    
    last_exception = None
    current_delay = delay
    
    for attempt in range(max_attempts):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            
            if attempt == max_attempts - 1:
                raise e
            
            time.sleep(current_delay)
            current_delay *= backoff_factor
    
    # This should never be reached, but just in case
    raise last_exception


def validate_email(email: str) -> bool:
    """Simple email validation."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_phone(phone: str, country: str = "SE") -> bool:
    """Simple phone number validation for Swedish numbers."""
    import re
    
    if country == "SE":
        # Swedish phone number patterns
        patterns = [
            r'^\+46\d{8,9}$',  # +46 followed by 8-9 digits
            r'^0\d{8,9}$',     # 0 followed by 8-9 digits
        ]
        return any(re.match(pattern, phone.replace(" ", "").replace("-", "")) for pattern in patterns)
    
    return False


def calculate_business_days(start_date: date, end_date: date) -> int:
    """Calculate number of business days between two dates."""
    import numpy as np
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Filter to business days (Monday=0, Sunday=6)
    business_days = date_range[date_range.weekday < 5]
    
    return len(business_days)


def get_business_day_offset(base_date: date, offset_days: int) -> date:
    """Get business day that is offset_days business days from base_date."""
    import pandas as pd
    
    if offset_days == 0:
        return base_date
    
    # Generate business days
    if offset_days > 0:
        date_range = pd.date_range(start=base_date, periods=offset_days * 2, freq='B')
        return date_range[offset_days].date()
    else:
        date_range = pd.date_range(end=base_date, periods=abs(offset_days) * 2, freq='B')
        return date_range[offset_days].date()


def create_summary_statistics(data: List[float]) -> Dict[str, float]:
    """Create summary statistics for a list of numbers."""
    if not data:
        return {}
    
    data_array = np.array(data)
    
    return {
        "count": len(data),
        "mean": float(np.mean(data_array)),
        "median": float(np.median(data_array)),
        "std": float(np.std(data_array)),
        "min": float(np.min(data_array)),
        "max": float(np.max(data_array)),
        "q25": float(np.percentile(data_array, 25)),
        "q75": float(np.percentile(data_array, 75)),
        "sum": float(np.sum(data_array))
    }


def normalize_fund_name(name: str) -> str:
    """Normalize fund name for consistent comparison."""
    # Remove common prefixes/suffixes and normalize case
    name = name.strip()
    
    # Common normalizations
    replacements = {
        " A ": " ",
        " SEK": "",
        " Index": " index",
        " indeks": " index", 
        " indeks ": " index ",
        "småb": "smab"
    }
    
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    # Clean up whitespace
    name = " ".join(name.split())
    
    return name


def generate_correlation_matrix(data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Generate correlation matrix as nested dictionary."""
    if data.empty:
        return {}
    
    corr_matrix = data.corr()
    
    # Convert to nested dict
    result = {}
    for col1 in corr_matrix.columns:
        result[col1] = {}
        for col2 in corr_matrix.columns:
            result[col1][col2] = float(corr_matrix.loc[col1, col2])
    
    return result


def calculate_rolling_metrics(
    data: pd.Series, 
    window: int,
    metrics: List[str] = None
) -> Dict[str, pd.Series]:
    """Calculate rolling metrics for time series data."""
    
    if metrics is None:
        metrics = ["mean", "std", "min", "max"]
    
    results = {}
    
    for metric in metrics:
        if metric == "mean":
            results["rolling_mean"] = data.rolling(window).mean()
        elif metric == "std":
            results["rolling_std"] = data.rolling(window).std()
        elif metric == "min":
            results["rolling_min"] = data.rolling(window).min()
        elif metric == "max":
            results["rolling_max"] = data.rolling(window).max()
        elif metric == "median":
            results["rolling_median"] = data.rolling(window).median()
    
    return results


def create_date_range_filter(
    start_date: Optional[Union[str, date, datetime]] = None,
    end_date: Optional[Union[str, date, datetime]] = None,
    days_back: Optional[int] = None
) -> Dict[str, date]:
    """Create standardized date range filter."""
    
    if days_back is not None:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
    else:
        if start_date is None:
            start_date = datetime.now().date() - timedelta(days=365)
        elif isinstance(start_date, str):
            start_date = parse_date(start_date).date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()
        
        if end_date is None:
            end_date = datetime.now().date()
        elif isinstance(end_date, str):
            end_date = parse_date(end_date).date()
        elif isinstance(end_date, datetime):
            end_date = end_date.date()
    
    return {
        "start_date": start_date,
        "end_date": end_date
    }


def save_json_file(data: Any, file_path: Union[str, Path], indent: int = 2) -> bool:
    """Save data as JSON file with error handling."""
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        
        return True
    except Exception:
        return False


def load_json_file(file_path: Union[str, Path]) -> Optional[Any]:
    """Load data from JSON file with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def get_file_age_days(file_path: Union[str, Path]) -> Optional[int]:
    """Get age of file in days."""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return None
        
        modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
        age = datetime.now() - modified_time
        return age.days
    except Exception:
        return None


class MemoryCache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, default_ttl: int = 3600):
        self.cache = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            value, expiry = self.cache[key]
            if datetime.now() < expiry:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        if ttl is None:
            ttl = self.default_ttl
        
        expiry = datetime.now() + timedelta(seconds=ttl)
        self.cache[key] = (value, expiry)
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
    
    def cleanup(self) -> int:
        """Remove expired entries and return count of removed items."""
        now = datetime.now()
        expired_keys = [
            key for key, (_, expiry) in self.cache.items()
            if now >= expiry
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)


# Global cache instance
_memory_cache = MemoryCache()


def get_cache() -> MemoryCache:
    """Get global memory cache instance."""
    return _memory_cache