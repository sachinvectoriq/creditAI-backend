from services.account_overview.main import account_app as account_overview_app
from services.AI_analysis.main import ai_analysis_app as ai_analysis_app
from services.financial_statement.main import financial_app as financial_statement_app
from fastapi import FastAPI

app = FastAPI(
    title="Credit AI Gateway",
    description="Gateway that forwards requests to the underlying microservices.",
    version="1.0.0",
)
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for the gateway."""
    return {
        "status": "healthy",
        "service": "credit_ai_gateway",
        "version": "1.0.0",
    }
# app.mount("/account-overview", account_overview_app)
# app.mount("/ai-analysis", ai_analysis_app)
# app.mount("/financial-statement", financial_statement_app)
