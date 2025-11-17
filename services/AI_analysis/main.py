from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from services.AI_analysis.api.endpoints import router as api_router


logging.basicConfig(
level=logging.INFO,
format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("ai-analysis-service")


ai_analysis_app = FastAPI(title="AI Analysis Service", version="1.0.0")


ai_analysis_app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)


ai_analysis_app.include_router(api_router)


# Uvicorn entry-point: `uvicorn main:ai_analysis_app --host 0.0.0.0 --port 8080`