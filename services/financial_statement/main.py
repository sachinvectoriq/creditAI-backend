# main.py
"""
Financial Statement Service - FastAPI Application
Fetches financial statements from NASDAQ API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import os

# ✅ CHANGE: use the in-package router to avoid fragile sys.path hacks
#    Original code attempted to import from "services.financial_statement..."
#    which breaks when running this microservice standalone. This import is
#    stable given the required structure (api/endpoints.py).
from services.financial_statement.api.endpoints import router as api_router


# ---------- Minimal settings / logging (safe defaults) ----------
class _Settings:
    # ✅ CHANGE: provide safe defaults if no external config is present
    environment: str = os.getenv("ENVIRONMENT", "local")
    api_v1_prefix: str = os.getenv("API_V1_PREFIX", "/api/v1")
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    cors_origins: list[str] = (
        os.getenv("CORS_ORIGINS", "*").split(",")
        if os.getenv("CORS_ORIGINS")
        else ["*"]
    )


def get_settings() -> _Settings:
    return _Settings()


def setup_logging(name: str = "creditai.financial_statement") -> logging.Logger:
    # ✅ CHANGE: simple, idempotent logger setup so logs appear in uvicorn
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, get_settings().log_level.upper(), logging.INFO))
    return logger
# ---------------------------------------------------------------


logger = setup_logging("creditai.financial_statement")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    Initialize resources on startup and cleanup on shutdown.
    """
    # Startup
    logger.info("Starting Financial Statement Service...")
    logger.info(f"Environment: {settings.environment}")
    yield
    # Shutdown
    logger.info("Shutting down Financial Statement Service...")


# Create FastAPI application
financial_app = FastAPI(
    title="Financial Statement Service",
    description="Microservice for fetching and processing financial statements from NASDAQ API",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
financial_app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@financial_app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": "financial_statement",
        "version": "1.0.0",
        "environment": settings.environment,
    }


# Include API routes
# ✅ CHANGE: prefix now comes from local settings (with default /api/v1)
financial_app.include_router(api_router, prefix=settings.api_v1_prefix)


# Global exception handler
@financial_app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions globally."""
    logger.error(f"HTTP error occurred: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@financial_app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions globally."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error occurred", "status_code": 500},
    )


if __name__ == "__main__":
    # ✅ CHANGE: uvicorn entry kept intact, driven by local settings
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
