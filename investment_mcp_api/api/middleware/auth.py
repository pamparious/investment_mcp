"""Authentication middleware for Investment MCP API."""

import time
import hashlib
import logging
from typing import Optional, Dict, Any
from uuid import uuid4

from fastapi import Request, HTTPException
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ..common.config import get_settings
from ..common.exceptions import AuthenticationException


logger = logging.getLogger(__name__)


class APIKeyManager:
    """Manage API keys and authentication."""
    
    def __init__(self):
        """Initialize API key manager."""
        self.settings = get_settings()
        
        # In production, this would be loaded from database
        self._api_keys = {
            "test-key-123": {
                "user_id": "test-user",
                "tier": "standard",
                "rate_limit": 100,
                "permissions": ["read", "write"],
                "created_at": time.time(),
                "last_used": None,
                "is_active": True
            },
            "premium-key-456": {
                "user_id": "premium-user", 
                "tier": "premium",
                "rate_limit": 1000,
                "permissions": ["read", "write", "admin"],
                "created_at": time.time(),
                "last_used": None,
                "is_active": True
            },
            "admin-key-789": {
                "user_id": "admin-user",
                "tier": "admin", 
                "rate_limit": 10000,
                "permissions": ["read", "write", "admin", "system"],
                "created_at": time.time(),
                "last_used": None,
                "is_active": True
            }
        }
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user information."""
        if not api_key:
            return None
        
        # Hash the API key for lookup (in production)
        # For demo purposes, we'll use the key directly
        key_info = self._api_keys.get(api_key)
        
        if not key_info:
            return None
        
        if not key_info.get("is_active", False):
            return None
        
        # Update last used timestamp
        key_info["last_used"] = time.time()
        
        return key_info
    
    def get_user_permissions(self, api_key: str) -> list:
        """Get user permissions for API key."""
        key_info = self.validate_api_key(api_key)
        if not key_info:
            return []
        
        return key_info.get("permissions", [])
    
    def get_rate_limit(self, api_key: str) -> int:
        """Get rate limit for API key."""
        key_info = self.validate_api_key(api_key)
        if not key_info:
            return 10  # Very low default rate limit
        
        return key_info.get("rate_limit", 100)
    
    def get_user_tier(self, api_key: str) -> str:
        """Get user tier for API key."""
        key_info = self.validate_api_key(api_key)
        if not key_info:
            return "free"
        
        return key_info.get("tier", "standard")


# Global API key manager instance
api_key_manager = APIKeyManager()


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication."""
    
    # Endpoints that don't require authentication
    EXEMPT_PATHS = {
        "/",
        "/health",
        "/docs",
        "/redoc", 
        "/openapi.json",
        "/api/v1/openapi.json"
    }
    
    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()
    
    async def dispatch(self, request: Request, call_next):
        """Process request authentication."""
        
        # Skip authentication for exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)
        
        # Skip authentication for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Extract API key from header
        api_key = request.headers.get(self.settings.API_KEY_HEADER)
        
        if not api_key:
            logger.warning(f"Missing API key for {request.url.path}")
            raise AuthenticationException("API key required")
        
        # Validate API key
        key_info = api_key_manager.validate_api_key(api_key)
        if not key_info:
            logger.warning(f"Invalid API key attempted: {api_key[:10]}...")
            raise AuthenticationException("Invalid API key")
        
        # Add user information to request state
        request.state.user_id = key_info["user_id"]
        request.state.user_tier = key_info["tier"]
        request.state.user_permissions = key_info["permissions"]
        request.state.rate_limit = key_info["rate_limit"]
        request.state.api_key = api_key
        
        # Generate request ID for tracking
        request.state.request_id = str(uuid4())
        
        # Log authentication success
        logger.info(
            f"Authenticated user {key_info['user_id']} "
            f"(tier: {key_info['tier']}) for {request.method} {request.url.path}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request.state.request_id
        
        return response


def get_current_user(request: Request) -> Dict[str, Any]:
    """Get current authenticated user information."""
    return {
        "user_id": getattr(request.state, "user_id", None),
        "tier": getattr(request.state, "user_tier", "free"),
        "permissions": getattr(request.state, "user_permissions", []),
        "rate_limit": getattr(request.state, "rate_limit", 10),
        "request_id": getattr(request.state, "request_id", None)
    }


def require_permission(permission: str):
    """Decorator to require specific permission."""
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            user_permissions = getattr(request.state, "user_permissions", [])
            if permission not in user_permissions:
                raise AuthenticationException(f"Permission '{permission}' required")
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


def require_tier(min_tier: str):
    """Decorator to require minimum user tier."""
    tier_hierarchy = {"free": 0, "standard": 1, "premium": 2, "admin": 3}
    
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            user_tier = getattr(request.state, "user_tier", "free")
            min_tier_level = tier_hierarchy.get(min_tier, 999)
            user_tier_level = tier_hierarchy.get(user_tier, 0)
            
            if user_tier_level < min_tier_level:
                raise AuthenticationException(f"Tier '{min_tier}' or higher required")
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


class APIKeyHeader(APIKeyHeader):
    """Custom API key header handler."""
    
    def __init__(self):
        super().__init__(name=get_settings().API_KEY_HEADER, auto_error=False)
    
    async def __call__(self, request: Request) -> Optional[str]:
        """Extract API key from request."""
        api_key = request.headers.get(self.model.name)
        
        if not api_key and self.auto_error:
            raise HTTPException(
                status_code=401,
                detail="API key required",
                headers={"WWW-Authenticate": "ApiKey"}
            )
        
        return api_key


# Security scheme for OpenAPI documentation
api_key_header = APIKeyHeader()


def generate_api_key(user_id: str, tier: str = "standard") -> str:
    """Generate a new API key for a user."""
    timestamp = str(int(time.time()))
    unique_data = f"{user_id}:{tier}:{timestamp}:{uuid4()}"
    
    # Create hash
    hash_object = hashlib.sha256(unique_data.encode())
    api_key = hash_object.hexdigest()[:32]
    
    return f"{tier}-{api_key}"


def revoke_api_key(api_key: str) -> bool:
    """Revoke an API key."""
    if api_key in api_key_manager._api_keys:
        api_key_manager._api_keys[api_key]["is_active"] = False
        logger.info(f"API key revoked: {api_key[:10]}...")
        return True
    return False


def list_user_api_keys(user_id: str) -> List[Dict[str, Any]]:
    """List all API keys for a user."""
    user_keys = []
    
    for api_key, key_info in api_key_manager._api_keys.items():
        if key_info["user_id"] == user_id:
            # Don't include the actual key in the response
            safe_key_info = {
                "key_id": api_key[:10] + "...",
                "tier": key_info["tier"],
                "created_at": key_info["created_at"],
                "last_used": key_info["last_used"],
                "is_active": key_info["is_active"],
                "permissions": key_info["permissions"]
            }
            user_keys.append(safe_key_info)
    
    return user_keys


def audit_api_key_usage(api_key: str, endpoint: str, status_code: int):
    """Audit API key usage for monitoring."""
    # In production, this would log to a monitoring system
    logger.info(
        f"API key usage: {api_key[:10]}... | "
        f"Endpoint: {endpoint} | "
        f"Status: {status_code}"
    )