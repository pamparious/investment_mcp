"""Rate limiting middleware for Investment MCP API."""

import time
import asyncio
import logging
from typing import Dict, Any, Optional
from collections import defaultdict, deque

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..common.config import get_settings
from ..common.exceptions import RateLimitException


logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket rate limiter implementation."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        async with self._lock:
            now = time.time()
            
            # Add tokens based on time elapsed
            time_passed = now - self.last_refill
            tokens_to_add = time_passed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait for tokens to be available."""
        if self.tokens >= tokens:
            return 0.0
        
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class SlidingWindowCounter:
    """Sliding window rate limiter implementation."""
    
    def __init__(self, window_size: int, max_requests: int):
        """
        Initialize sliding window counter.
        
        Args:
            window_size: Time window in seconds
            max_requests: Maximum requests in window
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()
        self._lock = asyncio.Lock()
    
    async def is_allowed(self) -> tuple[bool, Optional[float]]:
        """
        Check if request is allowed.
        
        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        async with self._lock:
            now = time.time()
            
            # Remove old requests outside window
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            
            # Check if under limit
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True, None
            
            # Calculate retry after time
            oldest_request = self.requests[0]
            retry_after = oldest_request + self.window_size - now
            return False, max(retry_after, 1.0)


class RateLimiter:
    """Comprehensive rate limiter with multiple strategies."""
    
    def __init__(self):
        """Initialize rate limiter."""
        self.settings = get_settings()
        
        # Store rate limiters per user/IP
        self.user_limiters: Dict[str, TokenBucket] = {}
        self.ip_limiters: Dict[str, SlidingWindowCounter] = {}
        self.endpoint_limiters: Dict[str, Dict[str, SlidingWindowCounter]] = defaultdict(dict)
        
        # Global rate limiter for DDoS protection
        self.global_limiter = SlidingWindowCounter(
            window_size=60,
            max_requests=5000  # Global limit: 5000 requests per minute
        )
    
    async def check_rate_limit(
        self, 
        request: Request,
        user_id: Optional[str] = None,
        user_tier: str = "standard"
    ) -> tuple[bool, Optional[float], Dict[str, Any]]:
        """
        Check if request should be rate limited.
        
        Returns:
            Tuple of (is_allowed, retry_after, rate_limit_info)
        """
        client_ip = self._get_client_ip(request)
        endpoint = self._get_endpoint_key(request)
        
        # Check global rate limit first
        global_allowed, global_retry = await self.global_limiter.is_allowed()
        if not global_allowed:
            logger.warning(f"Global rate limit exceeded from IP {client_ip}")
            return False, global_retry, {"limit_type": "global", "client_ip": client_ip}
        
        # Check user-specific rate limit if authenticated
        if user_id:
            user_allowed, user_retry, user_info = await self._check_user_rate_limit(
                user_id, user_tier
            )
            if not user_allowed:
                return False, user_retry, user_info
        
        # Check IP-based rate limit
        ip_allowed, ip_retry, ip_info = await self._check_ip_rate_limit(client_ip)
        if not ip_allowed:
            return False, ip_retry, ip_info
        
        # Check endpoint-specific rate limit
        endpoint_allowed, endpoint_retry, endpoint_info = await self._check_endpoint_rate_limit(
            endpoint, user_id or client_ip
        )
        if not endpoint_allowed:
            return False, endpoint_retry, endpoint_info
        
        # All checks passed
        return True, None, {"status": "allowed"}
    
    async def _check_user_rate_limit(
        self, 
        user_id: str, 
        user_tier: str
    ) -> tuple[bool, Optional[float], Dict[str, Any]]:
        """Check user-specific rate limit."""
        
        # Get rate limit based on user tier
        rate_limits = {
            "free": {"capacity": 10, "refill_rate": 10/60},      # 10 per minute
            "standard": {"capacity": 100, "refill_rate": 100/60}, # 100 per minute
            "premium": {"capacity": 1000, "refill_rate": 1000/60}, # 1000 per minute
            "admin": {"capacity": 10000, "refill_rate": 10000/60}  # 10000 per minute
        }
        
        limits = rate_limits.get(user_tier, rate_limits["standard"])
        
        # Get or create user rate limiter
        if user_id not in self.user_limiters:
            self.user_limiters[user_id] = TokenBucket(
                capacity=limits["capacity"],
                refill_rate=limits["refill_rate"]
            )
        
        bucket = self.user_limiters[user_id]
        allowed = await bucket.consume(1)
        
        if not allowed:
            retry_after = bucket.get_wait_time(1)
            return False, retry_after, {
                "limit_type": "user",
                "user_id": user_id,
                "user_tier": user_tier,
                "limit": limits["capacity"]
            }
        
        return True, None, {"user_id": user_id}
    
    async def _check_ip_rate_limit(
        self, 
        client_ip: str
    ) -> tuple[bool, Optional[float], Dict[str, Any]]:
        """Check IP-based rate limit."""
        
        # Create IP limiter if not exists
        if client_ip not in self.ip_limiters:
            self.ip_limiters[client_ip] = SlidingWindowCounter(
                window_size=60,  # 1 minute window
                max_requests=200  # 200 requests per minute per IP
            )
        
        limiter = self.ip_limiters[client_ip]
        allowed, retry_after = await limiter.is_allowed()
        
        if not allowed:
            return False, retry_after, {
                "limit_type": "ip",
                "client_ip": client_ip,
                "limit": 200
            }
        
        return True, None, {"client_ip": client_ip}
    
    async def _check_endpoint_rate_limit(
        self, 
        endpoint: str, 
        identifier: str
    ) -> tuple[bool, Optional[float], Dict[str, Any]]:
        """Check endpoint-specific rate limit."""
        
        # Endpoint-specific limits
        endpoint_limits = {
            "/api/v1/portfolio/analysis": {"window": 60, "max_requests": 10},
            "/api/v1/ai/investment-recommendation": {"window": 300, "max_requests": 5},
            "/api/v1/portfolio/stress-test": {"window": 60, "max_requests": 20},
            "/api/v1/funds/*/historical": {"window": 60, "max_requests": 30},
        }
        
        # Check if endpoint has specific limits
        limit_config = None
        for pattern, config in endpoint_limits.items():
            if self._endpoint_matches(endpoint, pattern):
                limit_config = config
                break
        
        if not limit_config:
            return True, None, {"endpoint": endpoint}
        
        # Get or create endpoint limiter
        if identifier not in self.endpoint_limiters[endpoint]:
            self.endpoint_limiters[endpoint][identifier] = SlidingWindowCounter(
                window_size=limit_config["window"],
                max_requests=limit_config["max_requests"]
            )
        
        limiter = self.endpoint_limiters[endpoint][identifier]
        allowed, retry_after = await limiter.is_allowed()
        
        if not allowed:
            return False, retry_after, {
                "limit_type": "endpoint",
                "endpoint": endpoint,
                "limit": limit_config["max_requests"],
                "window": limit_config["window"]
            }
        
        return True, None, {"endpoint": endpoint}
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded IP headers (for load balancers/proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to client IP
        return request.client.host if request.client else "unknown"
    
    def _get_endpoint_key(self, request: Request) -> str:
        """Get normalized endpoint key for rate limiting."""
        path = request.url.path
        
        # Normalize paths with parameters
        if "/funds/" in path and path.count("/") > 3:
            # Convert /api/v1/funds/FUND_CODE/... to /api/v1/funds/*/...
            parts = path.split("/")
            if len(parts) > 4 and parts[3] == "funds":
                parts[4] = "*"
                path = "/".join(parts)
        
        return f"{request.method}:{path}"
    
    def _endpoint_matches(self, endpoint: str, pattern: str) -> bool:
        """Check if endpoint matches pattern with wildcards."""
        endpoint_parts = endpoint.split(":")
        pattern_parts = pattern.split(":")
        
        if len(endpoint_parts) != len(pattern_parts):
            return False
        
        # Check method
        if pattern_parts[0] != "*" and endpoint_parts[0] != pattern_parts[0]:
            return False
        
        # Check path with wildcards
        endpoint_path = endpoint_parts[1]
        pattern_path = pattern_parts[1] if len(pattern_parts) > 1 else ""
        
        return self._path_matches(endpoint_path, pattern_path)
    
    def _path_matches(self, path: str, pattern: str) -> bool:
        """Check if path matches pattern with wildcards."""
        if "*" not in pattern:
            return path == pattern
        
        path_parts = path.split("/")
        pattern_parts = pattern.split("/")
        
        if len(path_parts) != len(pattern_parts):
            return False
        
        for path_part, pattern_part in zip(path_parts, pattern_parts):
            if pattern_part != "*" and path_part != pattern_part:
                return False
        
        return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app):
        super().__init__(app)
        self.rate_limiter = RateLimiter()
        self.settings = get_settings()
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        
        # Skip rate limiting for health checks and docs
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Get user info from request state (set by auth middleware)
        user_id = getattr(request.state, "user_id", None)
        user_tier = getattr(request.state, "user_tier", "free")
        
        # Check rate limits
        try:
            allowed, retry_after, limit_info = await self.rate_limiter.check_rate_limit(
                request, user_id, user_tier
            )
            
            if not allowed:
                logger.warning(f"Rate limit exceeded: {limit_info}")
                
                # Create rate limit response
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "retry_after": retry_after,
                        "limit_info": limit_info,
                        "request_id": getattr(request.state, "request_id", None)
                    },
                    headers={
                        "Retry-After": str(int(retry_after)) if retry_after else "60",
                        "X-RateLimit-Limit": str(limit_info.get("limit", "unknown")),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(time.time() + (retry_after or 60)))
                    }
                )
            
            # Add rate limit info to request state
            request.state.rate_limit_info = limit_info
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # On rate limiter error, allow request but log the issue
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        if hasattr(request.state, "rate_limit_info"):
            info = request.state.rate_limit_info
            if "limit" in info:
                response.headers["X-RateLimit-Limit"] = str(info["limit"])
                response.headers["X-RateLimit-Remaining"] = str(info.get("remaining", "unknown"))
        
        return response


def get_rate_limit_status(user_id: str) -> Dict[str, Any]:
    """Get current rate limit status for a user."""
    # This would typically query the rate limiter state
    return {
        "user_id": user_id,
        "current_usage": "unknown",
        "limit": "unknown",
        "reset_time": "unknown",
        "status": "This endpoint would provide real rate limit status"
    }