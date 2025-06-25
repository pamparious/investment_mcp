"""Logging middleware for Investment MCP API."""

import time
import json
import logging
from typing import Any, Dict
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ..common.config import get_settings


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request/response logging."""
    
    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()
        self.logger = logging.getLogger("api.requests")
        
        # Configure structured logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    async def dispatch(self, request: Request, call_next):
        """Log request and response details."""
        
        # Generate request ID if not exists
        if not hasattr(request.state, "request_id"):
            request.state.request_id = str(uuid4())
        
        # Record start time
        start_time = time.time()
        
        # Extract request info
        request_info = await self._extract_request_info(request)
        
        # Log request start
        self.logger.info(f"Request started", extra={
            "event": "request_start",
            "request_id": request.state.request_id,
            **request_info
        })
        
        # Process request
        response = None
        error = None
        
        try:
            response = await call_next(request)
        except Exception as e:
            error = e
            self.logger.error(f"Request failed with exception", extra={
                "event": "request_error",
                "request_id": request.state.request_id,
                "error": str(e),
                "error_type": type(e).__name__,
                **request_info
            })
            raise
        finally:
            # Calculate duration
            duration = time.time() - start_time
            
            # Extract response info
            response_info = self._extract_response_info(response) if response else {}
            
            # Log request completion
            log_level = logging.INFO
            if error:
                log_level = logging.ERROR
            elif response and response.status_code >= 400:
                log_level = logging.WARNING
            
            self.logger.log(log_level, f"Request completed", extra={
                "event": "request_complete",
                "request_id": request.state.request_id,
                "duration_ms": round(duration * 1000, 2),
                "success": error is None and (not response or response.status_code < 400),
                **request_info,
                **response_info
            })
        
        return response
    
    async def _extract_request_info(self, request: Request) -> Dict[str, Any]:
        """Extract relevant information from request."""
        
        # Basic request info
        info = {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("User-Agent", ""),
            "content_type": request.headers.get("Content-Type", ""),
            "content_length": request.headers.get("Content-Length", 0)
        }
        
        # Add user info if available
        if hasattr(request.state, "user_id"):
            info["user_id"] = request.state.user_id
            info["user_tier"] = getattr(request.state, "user_tier", "unknown")
        
        # Add API key info (masked)
        api_key = request.headers.get(self.settings.API_KEY_HEADER)
        if api_key:
            info["api_key_prefix"] = api_key[:8] + "..."
        
        # Add request body info for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Only log body size for security
                body = await request.body()
                if body:
                    info["body_size"] = len(body)
                    
                    # For JSON requests, log field names (not values)
                    if "application/json" in info.get("content_type", ""):
                        try:
                            json_data = json.loads(body)
                            if isinstance(json_data, dict):
                                info["body_fields"] = list(json_data.keys())
                        except:
                            pass
            except Exception:
                # If we can't read body, that's okay
                pass
        
        return info
    
    def _extract_response_info(self, response: Response) -> Dict[str, Any]:
        """Extract relevant information from response."""
        
        if not response:
            return {}
        
        info = {
            "status_code": response.status_code,
            "response_headers": dict(response.headers)
        }
        
        # Add response size if available
        content_length = response.headers.get("content-length")
        if content_length:
            info["response_size"] = int(content_length)
        
        return info
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request headers."""
        # Check forwarded headers first (for load balancers)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"


class PerformanceLogger:
    """Logger for performance metrics and monitoring."""
    
    def __init__(self):
        self.logger = logging.getLogger("api.performance")
        self.metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "total_response_time": 0.0,
            "endpoint_stats": {},
            "user_stats": {}
        }
    
    def log_request_metrics(
        self, 
        endpoint: str, 
        method: str,
        status_code: int,
        duration_ms: float,
        user_id: str = None
    ):
        """Log performance metrics for a request."""
        
        # Update global metrics
        self.metrics["total_requests"] += 1
        self.metrics["total_response_time"] += duration_ms
        
        if status_code >= 400:
            self.metrics["total_errors"] += 1
        
        # Update endpoint metrics
        endpoint_key = f"{method}:{endpoint}"
        if endpoint_key not in self.metrics["endpoint_stats"]:
            self.metrics["endpoint_stats"][endpoint_key] = {
                "requests": 0,
                "errors": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0
            }
        
        endpoint_stats = self.metrics["endpoint_stats"][endpoint_key]
        endpoint_stats["requests"] += 1
        endpoint_stats["total_time"] += duration_ms
        endpoint_stats["avg_time"] = endpoint_stats["total_time"] / endpoint_stats["requests"]
        endpoint_stats["min_time"] = min(endpoint_stats["min_time"], duration_ms)
        endpoint_stats["max_time"] = max(endpoint_stats["max_time"], duration_ms)
        
        if status_code >= 400:
            endpoint_stats["errors"] += 1
        
        # Update user metrics
        if user_id:
            if user_id not in self.metrics["user_stats"]:
                self.metrics["user_stats"][user_id] = {
                    "requests": 0,
                    "errors": 0,
                    "total_time": 0.0
                }
            
            user_stats = self.metrics["user_stats"][user_id]
            user_stats["requests"] += 1
            user_stats["total_time"] += duration_ms
            
            if status_code >= 400:
                user_stats["errors"] += 1
        
        # Log slow requests
        if duration_ms > 1000:  # More than 1 second
            self.logger.warning(f"Slow request detected", extra={
                "endpoint": endpoint_key,
                "duration_ms": duration_ms,
                "status_code": status_code,
                "user_id": user_id
            })
        
        # Log errors
        if status_code >= 500:
            self.logger.error(f"Server error", extra={
                "endpoint": endpoint_key,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "user_id": user_id
            })
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current performance metrics summary."""
        
        total_requests = self.metrics["total_requests"]
        
        if total_requests == 0:
            return {"message": "No requests processed yet"}
        
        avg_response_time = self.metrics["total_response_time"] / total_requests
        error_rate = (self.metrics["total_errors"] / total_requests) * 100
        
        # Get top endpoints by request count
        top_endpoints = sorted(
            self.metrics["endpoint_stats"].items(),
            key=lambda x: x[1]["requests"],
            reverse=True
        )[:10]
        
        # Get slowest endpoints
        slowest_endpoints = sorted(
            self.metrics["endpoint_stats"].items(),
            key=lambda x: x[1]["avg_time"],
            reverse=True
        )[:5]
        
        return {
            "total_requests": total_requests,
            "total_errors": self.metrics["total_errors"],
            "error_rate_percent": round(error_rate, 2),
            "avg_response_time_ms": round(avg_response_time, 2),
            "top_endpoints": [
                {
                    "endpoint": endpoint,
                    "requests": stats["requests"],
                    "avg_time_ms": round(stats["avg_time"], 2),
                    "error_rate": round((stats["errors"] / stats["requests"]) * 100, 2)
                }
                for endpoint, stats in top_endpoints
            ],
            "slowest_endpoints": [
                {
                    "endpoint": endpoint,
                    "avg_time_ms": round(stats["avg_time"], 2),
                    "requests": stats["requests"]
                }
                for endpoint, stats in slowest_endpoints
            ]
        }


# Global performance logger instance
performance_logger = PerformanceLogger()


class SecurityLogger:
    """Logger for security events and suspicious activities."""
    
    def __init__(self):
        self.logger = logging.getLogger("api.security")
    
    def log_auth_failure(
        self, 
        client_ip: str, 
        reason: str, 
        request_path: str,
        user_agent: str = ""
    ):
        """Log authentication failure."""
        self.logger.warning("Authentication failed", extra={
            "event": "auth_failure",
            "client_ip": client_ip,
            "reason": reason,
            "request_path": request_path,
            "user_agent": user_agent
        })
    
    def log_rate_limit_exceeded(
        self,
        client_ip: str,
        user_id: str,
        endpoint: str,
        limit_type: str
    ):
        """Log rate limit exceeded events."""
        self.logger.warning("Rate limit exceeded", extra={
            "event": "rate_limit_exceeded",
            "client_ip": client_ip,
            "user_id": user_id,
            "endpoint": endpoint,
            "limit_type": limit_type
        })
    
    def log_suspicious_activity(
        self,
        client_ip: str,
        activity_type: str,
        details: Dict[str, Any]
    ):
        """Log suspicious activities."""
        self.logger.warning("Suspicious activity detected", extra={
            "event": "suspicious_activity",
            "client_ip": client_ip,
            "activity_type": activity_type,
            **details
        })


# Global security logger instance
security_logger = SecurityLogger()