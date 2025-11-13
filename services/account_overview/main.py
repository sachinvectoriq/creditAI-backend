# main.py — app bootstrap (only mounts Account Overview; ignore any html_account_overview logic)
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.endpoints import router as account_router

logger = logging.getLogger("account_overview")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("🚀 Starting Account Overview microservice...")
    try:
        yield
    finally:
        logger.info("🛑 Shutting down Account Overview microservice...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Account Overview Service",
        version="1.0.1",
        description="Builds Account Overview tables from Item List & Payment History inputs.",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
        expose_headers=["*"],
        max_age=600,
    )

    app.include_router(account_router, prefix="")
    return app


app = create_app()
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
