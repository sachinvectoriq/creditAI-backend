from services.account_overview.main import account_app as account_overview_app
from services.AI_analysis.main import ai_analysis_app as ai_analysis_app
from services.financial_statement.main import financial_app as financial_statement_app
from fastapi import FastAPI
import os
from saml import saml_login, saml_callback, extract_token

main = FastAPI(
    title="Credit AI Gateway",
    description="Gateway that forwards requests to the underlying microservices.",
    version="1.0.0",
)

main.config["SAML_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saml")
main.config["SECRET_KEY"] = os.getenv('JWT_SECRET_KEY')  # Replac


# ---- SAML routes ----
@main.route('/saml/login')
async def login(): # Changed to async def
    return await saml_login(main.config["SAML_PATH"]) # Added await
 
@main.route('/saml/callback', methods=['POST'])
async def login_callback(): # Changed to async def
    return await saml_callback(main.config["SAML_PATH"]) # Added await
 
@main.route('/saml/token/extract', methods=['POST'])
async def func_get_data_from_token(): # Changed to async def
    return await extract_token() # Added await

@main.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for the gateway."""
    return {
        "status": "healthy",
        "service": "credit_ai_gateway",
        "version": "1.0.0",
    }


main.mount("/account-overview", account_overview_app)
main.mount("/ai-analysis", ai_analysis_app)
main.mount("/financial-statement", financial_statement_app)