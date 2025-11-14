from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any
from fastapi import UploadFile, File, Form

class AIAnalysisRequest(BaseModel):
    
    # Mode 1: Auto-fetch via ticker + mapping_json
    ticker: Optional[str] = Field(None, description="Stock ticker (e.g., AAPL) for auto-fetch mode.")
    mapping_json: Optional[str] = Field(
        "./company_tickers_exchange.json", description="Path to JSON file mapping ticker→CIK; located alongside the service or as provided."
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





class AIAnalysisInput(BaseModel):
    ticker: Optional[str] = None
    mapping_json: Optional[str] = None
    similarity_top_k: int = Field(5, ge=1, le=20)
    # file: Optional[UploadFile] = None

    class Config:
        arbitrary_types_allowed = True  # for UploadFile in Pydantic v1
        # Pydantic v2: model_config = {"arbitrary_types_allowed": True}

# Dependency that builds the model from multipart fields (NO Request object used)
async def parse_ai_form(
    # file: Optional[UploadFile] = File(default=None),
    ticker: Optional[str] = Form(default=None),
    mapping_json: Optional[str] = Form(default=None),
    similarity_top_k: int = Form(default=5),
) -> AIAnalysisInput:
    return AIAnalysisInput(
        ticker=(ticker or None),
        mapping_json=(mapping_json or None),
        similarity_top_k=similarity_top_k,
        # file=file,
    )
