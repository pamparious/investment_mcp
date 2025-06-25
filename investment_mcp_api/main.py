"""
Investment MCP API - Main FastAPI Application

This is the main entry point for the Investment MCP API, built following OpenAPI standards.
Provides comprehensive Swedish investment analysis with historical data and AI recommendations.
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.middleware.auth import AuthenticationMiddleware
from api.middleware.rate_limit import RateLimitMiddleware
from api.middleware.logging import LoggingMiddleware
from api.v1.routers import (
    portfolio_router,
    funds_router, 
    economic_router,
    market_router,
    ai_router,
    system_router
)
from api.common.exceptions import (
    APIException,
    ValidationException,
    RateLimitException,
    AuthenticationException
)
from api.common.config import get_settings
from api.services.data.database import init_database, close_database
from api.services.data.cache import init_cache, close_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    print("🚀 Starting Investment MCP API...")
    
    # Initialize database
    await init_database()
    print("✅ Database initialized")
    
    # Initialize cache
    await init_cache()
    print("✅ Cache initialized")
    
    # Warm up AI models
    from api.services.analysis.ai_service import warm_up_ai_models
    await warm_up_ai_models()
    print("✅ AI models warmed up")
    
    print("🎯 Investment MCP API ready!")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Investment MCP API...")
    await close_cache()
    await close_database()
    print("✅ Cleanup completed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    settings = get_settings()
    
    # Create FastAPI app with custom OpenAPI
    app = FastAPI(
        title="Investment MCP API",
        description="""
        Swedish Investment Analysis API with Historical Data and AI-Powered Recommendations
        
        ## Features
        - 🇸🇪 Swedish market focus with 20+ years of historical data
        - 🤖 AI-powered portfolio optimization and analysis  
        - 📊 Real-time Swedish economic indicators (Riksbank, SCB)
        - 🎯 Risk assessment and stress testing
        - 📈 Market regime analysis and sentiment tracking
        
        ## Quick Start
        1. Obtain an API key from your administrator
        2. Include it in the `X-API-Key` header
        3. Start with `/health` to verify connectivity
        4. Use `/portfolio/analysis` for AI-powered recommendations
        
        ## Rate Limits
        - Standard: 100 requests/minute
        - Premium: 1000 requests/minute
        """,
        version="3.0.0",
        docs_url=None,  # Custom docs handling
        redoc_url=None,  # Custom redoc handling
        openapi_url="/api/v1/openapi.json",
        lifespan=lifespan,
        contact={
            "name": "Investment MCP API Support",
            "email": "support@investment-mcp.com",
            "url": "https://docs.investment-mcp.com"
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT"
        }
    )
    
    # Add middleware
    setup_middleware(app, settings)
    
    # Add routers
    setup_routers(app)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    # Custom OpenAPI schema
    setup_custom_openapi(app)
    
    return app


def setup_middleware(app: FastAPI, settings):
    """Configure application middleware."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Trusted hosts
    if settings.ENVIRONMENT == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS
        )
    
    # Custom middleware
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(AuthenticationMiddleware)


def setup_routers(app: FastAPI):
    """Configure API routers."""
    
    # API v1 routes
    app.include_router(
        portfolio_router,
        prefix="/api/v1/portfolio",
        tags=["Portfolio Analysis"]
    )
    
    app.include_router(
        funds_router,
        prefix="/api/v1/funds",
        tags=["Fund Data"]
    )
    
    app.include_router(
        economic_router,
        prefix="/api/v1/economic",
        tags=["Swedish Economic Data"]
    )
    
    app.include_router(
        market_router,
        prefix="/api/v1/market",
        tags=["Market Analysis"]
    )
    
    app.include_router(
        ai_router,
        prefix="/api/v1/ai",
        tags=["AI Analysis"]
    )
    
    app.include_router(
        system_router,
        prefix="/api/v1",
        tags=["System"]
    )


def setup_exception_handlers(app: FastAPI):
    """Configure custom exception handlers."""
    
    @app.exception_handler(APIException)
    async def api_exception_handler(request: Request, exc: APIException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "type": exc.error_type,
                "request_id": getattr(request.state, "request_id", None)
            }
        )
    
    @app.exception_handler(ValidationException)
    async def validation_exception_handler(request: Request, exc: ValidationException):
        return JSONResponse(
            status_code=400,
            content={
                "error": "Validation failed",
                "details": exc.errors,
                "request_id": getattr(request.state, "request_id", None)
            }
        )
    
    @app.exception_handler(RateLimitException)
    async def rate_limit_exception_handler(request: Request, exc: RateLimitException):
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "retry_after": exc.retry_after,
                "request_id": getattr(request.state, "request_id", None)
            },
            headers={"Retry-After": str(exc.retry_after)}
        )
    
    @app.exception_handler(AuthenticationException)
    async def auth_exception_handler(request: Request, exc: AuthenticationException):
        return JSONResponse(
            status_code=401,
            content={
                "error": "Authentication failed",
                "message": exc.detail,
                "request_id": getattr(request.state, "request_id", None)
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "request_id": getattr(request.state, "request_id", None)
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        import traceback
        import logging
        
        logger = logging.getLogger("api.error")
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "request_id": getattr(request.state, "request_id", None),
                "type": "internal_error"
            }
        )


def setup_custom_openapi(app: FastAPI):
    """Setup custom OpenAPI schema and documentation."""
    
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        
        # Add custom fields
        openapi_schema["info"]["x-api-id"] = "investment-mcp-api"
        openapi_schema["info"]["x-audience"] = "financial-advisors,individual-investors"
        
        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key for authentication"
            }
        }
        
        # Add global security requirement
        openapi_schema["security"] = [{"ApiKeyAuth": []}]
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    # Custom documentation endpoints
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - Swagger UI",
            swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.15.5/swagger-ui-bundle.js",
            swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4.15.5/swagger-ui.css",
        )
    
    @app.get("/redoc", include_in_schema=False)
    async def redoc_html():
        return get_redoc_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - ReDoc",
            redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2.0.0/bundles/redoc.standalone.js",
        )


# Create the app instance
app = create_app()


# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Investment MCP API",
        "version": "3.0.0",
        "description": "Swedish Investment Analysis API with AI-Powered Recommendations",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": "/api/v1/openapi.json",
        "health_check": "/api/v1/health",
        "status": "online"
    }


if __name__ == "__main__":
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
        server_header=False,
        date_header=False
    )