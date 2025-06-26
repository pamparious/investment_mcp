"""
Unified utilities package for Investment MCP System.

This package provides common utility functions and classes used throughout
the Investment MCP system for logging, data processing, and helper functions.
"""

from .logging import (
    setup_logging,
    get_logger,
    get_default_logger,
    log_function_call,
    log_async_function_call,
    PerformanceLogger,
    DataCollectionLogger,
    SecurityLogger
)

from .helpers import (
    safe_float,
    safe_int,
    safe_percentage,
    format_currency,
    format_percentage,
    format_date,
    parse_date,
    calculate_hash,
    chunk_list,
    flatten_dict,
    unflatten_dict,
    remove_none_values,
    merge_dicts,
    retry_async,
    retry_sync,
    validate_email,
    validate_phone,
    calculate_business_days,
    get_business_day_offset,
    create_summary_statistics,
    normalize_fund_name,
    generate_correlation_matrix,
    calculate_rolling_metrics,
    create_date_range_filter,
    save_json_file,
    load_json_file,
    get_file_age_days,
    MemoryCache,
    get_cache
)

__all__ = [
    # Logging utilities
    'setup_logging',
    'get_logger',
    'get_default_logger',
    'log_function_call',
    'log_async_function_call',
    'PerformanceLogger',
    'DataCollectionLogger',
    'SecurityLogger',
    
    # Helper functions
    'safe_float',
    'safe_int', 
    'safe_percentage',
    'format_currency',
    'format_percentage',
    'format_date',
    'parse_date',
    'calculate_hash',
    'chunk_list',
    'flatten_dict',
    'unflatten_dict',
    'remove_none_values',
    'merge_dicts',
    'retry_async',
    'retry_sync',
    'validate_email',
    'validate_phone',
    'calculate_business_days',
    'get_business_day_offset',
    'create_summary_statistics',
    'normalize_fund_name',
    'generate_correlation_matrix',
    'calculate_rolling_metrics',
    'create_date_range_filter',
    'save_json_file',
    'load_json_file',
    'get_file_age_days',
    'MemoryCache',
    'get_cache'
]