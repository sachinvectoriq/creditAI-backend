# endpoints.py — FastAPI router with API endpoints
from typing import List
import logging
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from services.account_overview.model import AccountOverviewRequestJSON, AccountOverviewResponse, MetaSummary
from services.account_overview.core.account_processor import load_inputs, build_account_overview, format_overview_table

logger = logging.getLogger("account_overview.endpoints")

router = APIRouter(tags=["Account Overview"])


@router.get("/healthz", summary="Liveness probe")
async def healthz():
    return {"status": "ok"}


def _read_table(file: UploadFile) -> pd.DataFrame:
    name = (file.filename or "").lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(file.file)
        return pd.read_excel(file.file, engine="openpyxl")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse '{file.filename}': {e}",
        )


def _require_columns(df: pd.DataFrame, needed: List[str], label: str):
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{label} missing required columns: {missing}",
        )


@router.post(
    "/account-overview/upload",
    response_model=AccountOverviewResponse,
    summary="Build Account Overview from uploaded files (multipart/form-data)",
)
async def account_overview_upload(
    item_list: UploadFile = File(..., description="item_list.xlsx or .csv"),
    payment_history: UploadFile = File(..., description="payment_history.xlsx or .csv"),
):
    try:
        items_df = _read_table(item_list)
        pay_df = _read_table(payment_history)

        # Match your sample sheets exactly
        _require_columns(items_df, ["Unit", "Days Late", "Item Balance"], "item_list")
        _require_columns(pay_df, ["Payment Date", "Amt Applied to Customer", "Days Past Due"], "payment_history")

        items_df, pay_df = load_inputs(items_df, pay_df)
        table_df, meta = build_account_overview(items_df, pay_df)
        table_json = format_overview_table(table_df)

        resp = AccountOverviewResponse(
            table=table_json,
            meta=MetaSummary.model_validate(meta),
        )
        return resp
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("account_overview_upload failed")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.post(
    "/account-overview/json",
    response_model=AccountOverviewResponse,
    summary="Build Account Overview from JSON payloads",
)
async def account_overview_json(payload: AccountOverviewRequestJSON):
    try:
        # Payload can mirror Excel columns exactly; extras are allowed by models
        items_df, pay_df = load_inputs(payload.item_list, payload.payment_history)
        table_df, meta = build_account_overview(items_df, pay_df)
        table_json = format_overview_table(table_df)

        resp = AccountOverviewResponse(
            table=table_json,
            meta=MetaSummary.model_validate(meta),
        )
        return resp
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("account_overview_json failed")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
