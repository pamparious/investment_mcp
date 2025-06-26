"""
Unified logging configuration for Investment MCP System.

This module provides centralized logging configuration for all components
of the Investment MCP system with consistent formatting and output.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
from ..core.config import get_settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True
) -> logging.Logger:
    """
    Setup unified logging configuration for the Investment MCP system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log file path (if None, uses default)
        enable_console: Enable console logging
        enable_file: Enable file logging
        
    Returns:
        Configured logger instance
    """
    
    settings = get_settings()
    
    # Set log level
    if log_level is None:
        log_level = settings.LOG_LEVEL
    
    # Create logger
    logger = logging.getLogger("investment_mcp")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if enable_file:
        if log_file is None:
            # Create logs directory
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"investment_mcp_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Rotating file handler (max 10MB, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    logger.info("Logging configured successfully")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Console logging: {enable_console}")
    logger.info(f"File logging: {enable_file}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module or component.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    
    return logging.getLogger(f"investment_mcp.{name}")


def log_function_call(func):
    """
    Decorator to log function calls with arguments and execution time.
    
    Usage:
        @log_function_call
        def my_function(arg1, arg2):
            return result
    """
    
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function call
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper


def log_async_function_call(func):
    """
    Decorator to log async function calls with arguments and execution time.
    
    Usage:
        @log_async_function_call
        async def my_async_function(arg1, arg2):
            return result
    """
    
    import functools
    import time
    import asyncio
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function call
        logger.debug(f"Calling async {func.__name__} with args={args}, kwargs={kwargs}")
        
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"Async {func.__name__} completed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Async {func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper


class PerformanceLogger:
    """Logger for tracking performance metrics and timing."""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.timings = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        import time
        self.timings[operation] = time.time()
        self.logger.debug(f"Started timing: {operation}")
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        import time
        
        if operation not in self.timings:
            self.logger.warning(f"Timer not found for operation: {operation}")
            return 0.0
        
        duration = time.time() - self.timings[operation]
        del self.timings[operation]
        
        self.logger.info(f"Operation '{operation}' completed in {duration:.3f}s")
        return duration
    
    def log_memory_usage(self, operation: str = "current"):
        """Log current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.logger.info(
                f"Memory usage for {operation}: "
                f"RSS={memory_info.rss / 1024 / 1024:.1f}MB, "
                f"VMS={memory_info.vms / 1024 / 1024:.1f}MB"
            )
        except ImportError:
            self.logger.debug("psutil not available for memory monitoring")
    
    def log_system_resources(self):
        """Log system resource usage."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            self.logger.info(
                f"System resources: "
                f"CPU={cpu_percent}%, "
                f"Memory={memory.percent}% ({memory.used / 1024 / 1024 / 1024:.1f}GB used), "
                f"Disk={disk.percent}% ({disk.used / 1024 / 1024 / 1024:.1f}GB used)"
            )
        except ImportError:
            self.logger.debug("psutil not available for system monitoring")


class DataCollectionLogger:
    """Specialized logger for data collection operations."""
    
    def __init__(self, collector_name: str):
        self.logger = get_logger(f"collectors.{collector_name}")
        self.performance = PerformanceLogger(f"collectors.{collector_name}")
        self.collection_stats = {}
    
    def log_collection_start(self, operation: str, **kwargs):
        """Log the start of a data collection operation."""
        self.logger.info(f"Starting {operation} collection with parameters: {kwargs}")
        self.performance.start_timer(operation)
    
    def log_collection_progress(self, operation: str, current: int, total: int):
        """Log progress of data collection."""
        progress = (current / total) * 100 if total > 0 else 0
        self.logger.info(f"{operation} progress: {current}/{total} ({progress:.1f}%)")
    
    def log_collection_complete(
        self, 
        operation: str, 
        records_collected: int,
        records_failed: int = 0,
        **kwargs
    ):
        """Log completion of data collection operation."""
        duration = self.performance.end_timer(operation)
        
        success_rate = (records_collected / (records_collected + records_failed)) * 100 if (records_collected + records_failed) > 0 else 0
        rate = records_collected / duration if duration > 0 else 0
        
        self.logger.info(
            f"{operation} completed: "
            f"{records_collected} records collected, "
            f"{records_failed} failed, "
            f"success rate: {success_rate:.1f}%, "
            f"rate: {rate:.1f} records/sec"
        )
        
        # Store stats
        self.collection_stats[operation] = {
            "records_collected": records_collected,
            "records_failed": records_failed,
            "duration": duration,
            "success_rate": success_rate,
            "collection_rate": rate,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_collection_summary(self) -> dict:
        """Get summary of all collection operations."""
        return {
            "total_operations": len(self.collection_stats),
            "total_records": sum(s["records_collected"] for s in self.collection_stats.values()),
            "total_failures": sum(s["records_failed"] for s in self.collection_stats.values()),
            "total_duration": sum(s["duration"] for s in self.collection_stats.values()),
            "operations": self.collection_stats
        }


class SecurityLogger:
    """Specialized logger for security-related events."""
    
    def __init__(self):
        self.logger = get_logger("security")
    
    def log_api_access(self, endpoint: str, client_ip: str, user_agent: str = None):
        """Log API access attempts."""
        self.logger.info(f"API access: {endpoint} from {client_ip} (UA: {user_agent})")
    
    def log_authentication_attempt(self, success: bool, client_ip: str, details: str = None):
        """Log authentication attempts."""
        status = "SUCCESS" if success else "FAILED"
        self.logger.warning(f"Authentication {status} from {client_ip}: {details}")
    
    def log_rate_limit_exceeded(self, client_ip: str, endpoint: str, current_rate: int, limit: int):
        """Log rate limit violations."""
        self.logger.warning(
            f"Rate limit exceeded: {client_ip} hit {endpoint} "
            f"{current_rate} times (limit: {limit})"
        )
    
    def log_suspicious_activity(self, activity: str, client_ip: str, details: dict = None):
        """Log suspicious activities."""
        self.logger.error(f"Suspicious activity: {activity} from {client_ip}, details: {details}")


# Initialize default logging when module is imported
_default_logger = None

def get_default_logger() -> logging.Logger:
    """Get the default Investment MCP logger."""
    global _default_logger
    
    if _default_logger is None:
        _default_logger = setup_logging()
    
    return _default_logger