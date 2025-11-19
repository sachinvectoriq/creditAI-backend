from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
import logging

from services.AI_analysis.model import AIAnalysisResponse, HealthResponse
from services.AI_analysis.core.aianalysis import (
    extract_text_from_file,
    build_latest_10q_url_from_mapping,
    run_full_pipeline_from_url,
    run_full_pipeline_from_text,
)

router = APIRouter()
logger = logging.getLogger("ai-analysis-endpoints")

# Default path for mapping JSON (Windows-style as requested)
DEFAULT_MAPPING_JSON = r"./services/AI_analysis/api/company_tickers_exchange.json"

@router.get("/healthz", response_model=HealthResponse)
async def healthz():
    return HealthResponse(status="ok")

# -------- Auto-fetch (no file) --------
@router.post("/ai-analysis/auto", response_model=AIAnalysisResponse, status_code=status.HTTP_200_OK)
async def ai_analysis_auto(
    ticker: str = Form(...),
    mapping_json: str = Form(DEFAULT_MAPPING_JSON),
    similarity_top_k: int = Form(5),
):
    """
    Auto-fetch latest 10-Q via ticker + mapping_json.
    - ticker: required
    - mapping_json: optional (defaults to core\\company_tickers_exchange.json)
    - similarity_top_k: optional (defaults to 5)
    """
    try:
        url = build_latest_10q_url_from_mapping(ticker.strip().upper(), mapping_json.strip())
        if not url:
            raise HTTPException(status_code=404, detail="Could not resolve latest 10-Q URL for the given ticker.")

        results = await run_full_pipeline_from_url(url, similarity_top_k=similarity_top_k)
        if results and all(isinstance(v, str) and v.startswith("Error:") for v in results.values()):
            raise HTTPException(status_code=502, detail="Upstream agent execution failed for all sections.")
        return AIAnalysisResponse(status="success", results=results)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("/ai-analysis/auto failed")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

# -------- Manual upload (file) --------
@router.post("/ai-analysis/upload", response_model=AIAnalysisResponse, status_code=status.HTTP_200_OK)
async def ai_analysis_upload(
    file: UploadFile = File(...),
    similarity_top_k: int = Form(5),
):
    """
    Manual 10-Q upload (PDF/DOCX/TXT). File is required.
    - similarity_top_k: optional (defaults to 5)
    """
    try:
        contents = await file.read()
        text = extract_text_from_file(contents, filename=file.filename)
        if not text:
            raise HTTPException(status_code=400, detail="Unable to read uploaded file.")

        results = await run_full_pipeline_from_text(text, similarity_top_k=similarity_top_k)
        if results and all(isinstance(v, str) and v.startswith("Error:") for v in results.values()):
            raise HTTPException(status_code=502, detail="Upstream agent execution failed for all sections.")
        return AIAnalysisResponse(status="success", results=results)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("/ai-analysis/upload failed")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

