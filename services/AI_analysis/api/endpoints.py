from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi import status
from typing import Optional
import asyncio
import logging

from model import AIAnalysisRequest, AIAnalysisResponse, HealthResponse
from core.aianalysis import (
    extract_text_from_file,
    build_latest_10q_url_from_mapping,
    run_full_pipeline_from_url,
    run_full_pipeline_from_text,
)

router = APIRouter()
logger = logging.getLogger("ai-analysis-endpoints")

@router.get("/healthz", response_model=HealthResponse)
async def healthz():
    return HealthResponse(status="ok")

@router.post("/ai-analysis", response_model=AIAnalysisResponse, status_code=status.HTTP_200_OK)
async def ai_analysis(
    payload: AIAnalysisRequest = Depends(),
    file: Optional[UploadFile] = File(default=None),
):
    """Handle both manual upload and auto-fetch modes.

    - Manual: multipart/form-data with file field
    - Auto-fetch: JSON/form fields with ticker + mapping_json path
    """
    try:
        similarity_top_k = payload.similarity_top_k

        # Manual upload takes precedence if provided
        if file is not None:
            contents = await file.read()
            text = extract_text_from_file(contents, filename=file.filename)
            if not text:
                raise HTTPException(status_code=400, detail="Unable to read uploaded file.")
            results = await run_full_pipeline_from_text(text, similarity_top_k=similarity_top_k)
            # If all sections failed upstream, propagate a 502
            if results and all(isinstance(v, str) and v.startswith("Error:") for v in results.values()):
                raise HTTPException(status_code=502, detail="Upstream agent execution failed for all sections.")
            return AIAnalysisResponse(status="success", results=results)

        # Auto-fetch mode
        if not payload.ticker or not payload.mapping_json:
            raise HTTPException(
                status_code=400,
                detail="For auto-fetch mode, provide both 'ticker' and 'mapping_json'. Or upload a file.",
            )
        url = build_latest_10q_url_from_mapping(payload.ticker.upper(), payload.mapping_json)
        if not url:
            raise HTTPException(status_code=404, detail="Could not resolve latest 10-Q URL for the given ticker.")

        results = await run_full_pipeline_from_url(url, similarity_top_k=similarity_top_k)
        if results and all(isinstance(v, str) and v.startswith("Error:") for v in results.values()):
            raise HTTPException(status_code=502, detail="Upstream agent execution failed for all sections.")
        return AIAnalysisResponse(status="success", results=results)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("/ai-analysis failed")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")