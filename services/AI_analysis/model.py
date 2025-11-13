from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any

class AIAnalysisRequest(BaseModel):
    
    # Mode 1: Auto-fetch via ticker + mapping_json
    ticker: Optional[str] = Field(None, description="Stock ticker (e.g., AAPL) for auto-fetch mode.")
    mapping_json: Optional[str] = Field(
        None, description="Path to JSON file mapping ticker→CIK; located alongside the service or as provided."
    )

    # Mode 2: Manual upload via multipart/form-data file (handled in endpoint)
    # If provided, takes precedence over auto-fetch.

    # Optional query tuning
    similarity_top_k: int = Field(5, ge=1, le=20)

class AIAnalysisResponse(BaseModel):
    status: str
    results: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str