"""
Shared configuration module for all microservices.
Contains environment variables, Azure OpenAI settings, and common configurations.
"""
import os
from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # Application Settings
    app_name: str = "Financial Analysis Platform"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"  # development, staging, production
    
    # Azure OpenAI Configuration
    azure_openai_key: str
    azure_openai_endpoint: str
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_embedding_deployment: str = "text-embedding-ada-002"
    azure_openai_api_version: str = "2024-02-01"
    
    # API Configuration
    api_v1_prefix: str = "/api/v1"
    cors_origins: list = ["*"]
    
    # Service URLs (for inter-service communication)
    financial_service_url: Optional[str] = "http://localhost:8001"
    account_service_url: Optional[str] = "http://localhost:8002"
    ai_service_url: Optional[str] = "http://localhost:8003"
    
    # Database/Storage (if needed in future)
    redis_url: Optional[str] = None
    postgres_url: Optional[str] = None
    
    # File Upload Settings
    max_upload_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: list = ["xlsx", "xls", "csv", "pdf", "docx", "txt"]
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # json or text
    
    # Security
    secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 30
    
    # External APIs
    nasdaq_api_base_url: str = "https://api.nasdaq.com"
    sec_api_base_url: str = "https://data.sec.gov"
    sec_archive_base_url: str = "https://www.sec.gov/Archives/edgar/data"
    
    # Request Headers for External APIs
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Using lru_cache ensures we only create one instance.
    """
    return Settings()


# Common configuration constants
HEADERS = {
    "User-Agent": get_settings().user_agent,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

# Logging configuration
import logging
import sys


def setup_logging(service_name: str):
    """Setup logging configuration for a service"""
    settings = get_settings()
    
    log_level = getattr(logging, settings.log_level.upper())
    
    if settings.log_format == "json":
        import json
        
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_obj = {
                    "timestamp": self.formatTime(record, self.datefmt),
                    "service": service_name,
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                }
                if record.exc_info:
                    log_obj["exception"] = self.formatException(record.exc_info)
                return json.dumps(log_obj)
        
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            f"%(asctime)s - {service_name} - %(levelname)s - %(message)s"
        )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(handler)
    
    return logger


# Health check response model
from pydantic import BaseModel
from datetime import datetime


class HealthCheckResponse(BaseModel):
    """Standard health check response"""
    status: str = "healthy"
    service: str
    version: str
    timestamp: datetime
    environment: str