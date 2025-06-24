"""Rate limiting utilities for API calls."""

import asyncio
import time
from typing import Dict, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, max_calls: int, time_window: int):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make an API call."""
        async with self._lock:
            current_time = time.time()
            
            # Remove old calls outside the time window
            while self.calls and self.calls[0] <= current_time - self.time_window:
                self.calls.popleft()
            
            # Check if we can make a call
            if len(self.calls) < self.max_calls:
                self.calls.append(current_time)
                return
            
            # Calculate wait time
            wait_time = self.calls[0] + self.time_window - current_time + 0.1  # Small buffer
            logger.debug(f"Rate limit reached, waiting {wait_time:.2f} seconds")
            
            await asyncio.sleep(wait_time)
            
            # Try again after waiting
            await self.acquire()

class APIRateLimitManager:
    """Manage rate limits for multiple APIs."""
    
    def __init__(self):
        self.limiters: Dict[str, RateLimiter] = {}
    
    def add_limiter(self, api_name: str, max_calls: int, time_window: int):
        """Add a rate limiter for an API."""
        self.limiters[api_name] = RateLimiter(max_calls, time_window)
    
    async def acquire(self, api_name: str):
        """Acquire permission for an API call."""
        if api_name in self.limiters:
            await self.limiters[api_name].acquire()

# Global rate limit manager
rate_limit_manager = APIRateLimitManager()

# Add default limiters
rate_limit_manager.add_limiter("riksbank", 60, 60)  # 60 calls per minute
rate_limit_manager.add_limiter("scb", 30, 60)       # 30 calls per minute  
rate_limit_manager.add_limiter("yahoo_finance", 100, 60)  # 100 calls per minute