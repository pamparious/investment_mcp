"""API v1 routers for Investment MCP API."""

from .portfolio import router as portfolio_router
from .funds import router as funds_router
from .economic import router as economic_router
from .market import router as market_router
from .ai import router as ai_router
from .system import router as system_router


__all__ = [
    "portfolio_router",
    "funds_router", 
    "economic_router",
    "market_router",
    "ai_router",
    "system_router"
]