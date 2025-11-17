# api/endpoints.py
"""
API endpoints for Financial Statement Service
"""
from fastapi import APIRouter, HTTPException, Query, Path, BackgroundTasks
from typing import Optional, List, Dict, Any
import asyncio
import logging

# ✅ CHANGE: use local core import to match required structure
from services.financial_statement.core.financial_api import FinancialDataFetcher

# ---------- Local request/response models & helpers ----------
# ✅ CHANGE: Provide local pydantic models and enums so the service
#    runs independently without a separate 'shared.models' package.
from enum import Enum
from pydantic import BaseModel, Field, validator


class FrequencyType(str, Enum):
    ANNUAL = "annual"
    QUARTERLY = "quarterly"


def validate_ticker(ticker: str) -> str:
    """
    ✅ CHANGE: Lightweight ticker validation to prevent 4xx from upstream
    - Ensures alphabetic/period/hyphen chars and 1..10 length.
    """
    if not isinstance(ticker, str):
        raise HTTPException(status_code=400, detail="Ticker must be a string")
    t = ticker.strip().upper()
    if not (1 <= len(t) <= 10):
        raise HTTPException(status_code=400, detail="Ticker length invalid (1-10)")
    for ch in t:
        if not (ch.isalnum() or ch in ".-"):
            raise HTTPException(status_code=400, detail="Ticker contains invalid chars")
    return t


class FinancialStatementRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    frequency: FrequencyType = Field(
        FrequencyType.QUARTERLY, description="annual or quarterly"
    )

    # ✅ CHANGE: inline validation so bad tickers fail fast with 400
    @validator("ticker")
    def _v_ticker(cls, v: str) -> str:
        return validate_ticker(v)


class FinancialStatementResponse(BaseModel):
    success: bool
    data: Dict[str, Any] | None
    cached: bool = False
    message: Optional[str] = None


# Optional error envelope if you want to standardize errors in the future
class ErrorResponse(BaseModel):
    error: str
    status_code: int
# -------------------------------------------------------------

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Financial Statements"])


@router.post("/statements", response_model=FinancialStatementResponse)
async def get_financial_statements(
    request: FinancialStatementRequest, background_tasks: BackgroundTasks
) -> FinancialStatementResponse:
    """
    Fetch comprehensive financial statements for a ticker.
    """
    try:
        # Validate ticker (redundant but keeps explicit original flow)
        ticker = validate_ticker(request.ticker)

        # Create fetcher instance
        fetcher = FinancialDataFetcher(ticker)

        # ✅ CHANGE: map enum to API frequency (1=annual, 2=quarterly)
        frequency = 1 if request.frequency == FrequencyType.ANNUAL else 2

        # Fetch data asynchronously
        financial_data = await fetcher.fetch_all_statements_async(frequency)

        if not financial_data:
            raise HTTPException(
                status_code=404, detail=f"No financial data found for ticker {ticker}"
            )

        # Log successful fetch in background
        background_tasks.add_task(log_fetch_success, ticker, request.frequency.value)

        return FinancialStatementResponse(
            success=True,
            data=financial_data,
            cached=False,
            message=f"Successfully fetched financial statements for {ticker}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching financial statements: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch financial statements: {str(e)}"
        )


@router.get("/statements/{ticker}/income", response_model=Dict[str, Any])
async def get_income_statement(
    ticker: str = Path(..., description="Stock ticker symbol"),
    frequency: FrequencyType = Query(
        FrequencyType.QUARTERLY, description="Statement frequency"
    ),
) -> Dict[str, Any]:
    """
    Fetch income statement for a specific ticker.
    """
    try:
        ticker = validate_ticker(ticker)
        fetcher = FinancialDataFetcher(ticker)

        freq_num = 1 if frequency == FrequencyType.ANNUAL else 2
        income_statement_df = await fetcher.fetch_income_statement_async(freq_num)

        if income_statement_df is None or income_statement_df.empty:
            raise HTTPException(
                status_code=404, detail=f"Income statement not found for ticker {ticker}"
            )

        # ✅ CHANGE: serialize DataFrame to records so it’s JSON safe
        return {
            "ticker": ticker,
            "statement_type": "income_statement",
            "frequency": frequency.value,
            "data": income_statement_df.to_dict("records"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching income statement: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch income statement: {str(e)}"
        )


@router.get("/statements/{ticker}/balance", response_model=Dict[str, Any])
async def get_balance_sheet(
    ticker: str = Path(..., description="Stock ticker symbol"),
    frequency: FrequencyType = Query(
        FrequencyType.QUARTERLY, description="Statement frequency"
    ),
) -> Dict[str, Any]:
    """
    Fetch balance sheet for a specific ticker.
    """
    try:
        ticker = validate_ticker(ticker)
        fetcher = FinancialDataFetcher(ticker)

        freq_num = 1 if frequency == FrequencyType.ANNUAL else 2
        balance_sheet_df = await fetcher.fetch_balance_sheet_async(freq_num)

        if balance_sheet_df is None or balance_sheet_df.empty:
            raise HTTPException(
                status_code=404, detail=f"Balance sheet not found for ticker {ticker}"
            )

        # ✅ CHANGE: serialize DataFrame to records so it’s JSON safe
        return {
            "ticker": ticker,
            "statement_type": "balance_sheet",
            "frequency": frequency.value,
            "data": balance_sheet_df.to_dict("records"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching balance sheet: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch balance sheet: {str(e)}"
        )


@router.get("/statements/{ticker}/cashflow", response_model=Dict[str, Any])
async def get_cash_flow_statement(
    ticker: str = Path(..., description="Stock ticker symbol"),
    frequency: FrequencyType = Query(
        FrequencyType.QUARTERLY, description="Statement frequency"
    ),
) -> Dict[str, Any]:
    """
    Fetch cash flow statement for a specific ticker.
    """
    try:
        ticker = validate_ticker(ticker)
        fetcher = FinancialDataFetcher(ticker)

        freq_num = 1 if frequency == FrequencyType.ANNUAL else 2
        cash_flow_df = await fetcher.fetch_cash_flow_async(freq_num)

        if cash_flow_df is None or cash_flow_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Cash flow statement not found for ticker {ticker}",
            )

        # ✅ CHANGE: serialize DataFrame to records so it’s JSON safe
        return {
            "ticker": ticker,
            "statement_type": "cash_flow",
            "frequency": frequency.value,
            "data": cash_flow_df.to_dict("records"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching cash flow statement: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch cash flow statement: {str(e)}"
        )


@router.get("/statements/{ticker}/ratios", response_model=Dict[str, Any])
async def get_financial_ratios(
    ticker: str = Path(..., description="Stock ticker symbol"),
) -> Dict[str, Any]:
    """
    Calculate financial ratios from latest statements.
    """
    try:
        ticker = validate_ticker(ticker)
        fetcher = FinancialDataFetcher(ticker)

        # Fetch quarterly statements for ratio calculation
        financial_data = await fetcher.fetch_all_statements_async(frequency=2)

        if not financial_data:
            raise HTTPException(
                status_code=404, detail=f"No financial data found for ticker {ticker}"
            )

        ratios = await fetcher.calculate_financial_ratios_async(financial_data)

        return {
            "ticker": ticker,
            "ratios": ratios,
            "calculated_at": financial_data.get("metadata", {}).get("last_updated"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating financial ratios: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to calculate financial ratios: {str(e)}"
        )


@router.post("/statements/batch", response_model=Dict[str, Any])
async def get_batch_financial_statements(
    tickers: List[str],
    frequency: FrequencyType = FrequencyType.QUARTERLY,
) -> Dict[str, Any]:
    """
    Fetch financial statements for multiple tickers in parallel.
    """
    try:
        if len(tickers) > 10:
            raise HTTPException(
                status_code=400, detail="Maximum 10 tickers allowed per batch request"
            )

        # Validate all tickers
        validated_tickers = [validate_ticker(t) for t in tickers]

        # Fetch data in parallel
        freq_num = 1 if frequency == FrequencyType.ANNUAL else 2
        tasks = []
        for ticker in validated_tickers:
            fetcher = FinancialDataFetcher(ticker)
            tasks.append(fetcher.fetch_all_statements_async(freq_num))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        batch_results: Dict[str, Any] = {}
        errors: List[Dict[str, str]] = []

        for ticker, result in zip(validated_tickers, results):
            if isinstance(result, Exception):
                errors.append({"ticker": ticker, "error": str(result)})
                batch_results[ticker] = None
            else:
                batch_results[ticker] = result

        return {
            "success": len(errors) == 0,
            "results": batch_results,
            "errors": errors if errors else None,
            "total_requested": len(validated_tickers),
            "successful": len([r for r in batch_results.values() if r is not None]),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to process batch request: {str(e)}"
        )


async def log_fetch_success(ticker: str, frequency: str):
    """Background task to log successful fetches"""
    logging.getLogger(__name__).info(
        f"Successfully fetched {frequency} statements for {ticker}"
    )
