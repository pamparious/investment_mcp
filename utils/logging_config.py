"""Logging configuration for the investment MCP project."""

import logging
import logging.handlers
import os
from datetime import datetime
from config.settings import settings

def setup_logging():
    """Setup comprehensive logging for the application."""
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if not settings.DEBUG else logging.DEBUG)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for general logs
    file_handler = logging.handlers.RotatingFileHandler(
        filename=f"{log_dir}/investment_mcp.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Separate handler for data collection logs
    data_handler = logging.handlers.RotatingFileHandler(
        filename=f"{log_dir}/data_collection.log",
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    data_handler.setLevel(logging.INFO)
    data_handler.setFormatter(detailed_formatter)
    
    # Add data handler to data collection loggers
    data_logger = logging.getLogger("backend.data_collectors")
    data_logger.addHandler(data_handler)
    
    # Error handler for critical errors
    error_handler = logging.handlers.RotatingFileHandler(
        filename=f"{log_dir}/errors.log",
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    logging.info("Logging configuration complete")

if __name__ == "__main__":
    setup_logging()
    logging.info("Test log message")