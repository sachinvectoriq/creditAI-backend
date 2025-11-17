from services.account_overview.main import account_app as account_overview_app
from services.AI_analysis.main import ai_analysis_app as ai_analysis_app
from services.financial_statement.main import financial_app as financial_statement_app
from fastapi import FastAPI

main = FastAPI(
    title="Credit AI Gateway",
    description="Gateway that forwards requests to the underlying microservices.",
    version="1.0.0",
)
# @main.get("/health", tags=["Health"])
# async def health_check():
#     """Health check endpoint for the gateway."""
#     return {
#         "status": "healthy",
#         "service": "credit_ai_gateway",
#         "version": "1.0.0",
#     }
main.mount("/account-overview", account_overview_app)
main.mount("/ai-analysis", ai_analysis_app)
main.mount("/financial-statement", financial_statement_app)