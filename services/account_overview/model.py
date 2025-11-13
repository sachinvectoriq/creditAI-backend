# model.py — Pydantic request/response models & enums
from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator


class AgingBucket(str, Enum):
    CURRENT = "Current"
    D1_30 = "1-30"
    D31_60 = "31-60"
    D61_90 = "61-90"
    D91_180 = "91-180"
    D181P = "181+"
    TOTAL = "Total"


# ── Item List: your Excel has 3 columns: Unit, Days Late, Item Balance
class ItemListRow(BaseModel):
    Unit: str = Field(..., description="Business unit code (e.g., OCS01, OAV01)")
    Days_Late: int = Field(..., alias="Days Late")
    Item_Balance: Decimal = Field(..., alias="Item Balance")

    # Accept any extra columns (keeps JSON == Excel without validation noise)
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    @field_validator("Unit")
    @classmethod
    def _strip_unit(cls, v: str) -> str:
        return v.strip()

    @field_validator("Item_Balance")
    @classmethod
    def _ensure_decimal(cls, v: Decimal) -> Decimal:
        # Normalize to 2dp for deterministic results
        return Decimal(str(v)).quantize(Decimal("0.01"))


# ── Payment History: accept your sheet’s columns; only a subset is required
class PaymentHistoryRow(BaseModel):
    Payment_Date: Optional[date] = Field(None, alias="Payment Date")
    Amt_Applied_to_Customer: Optional[Decimal] = Field(None, alias="Amt Applied to Customer")
    Terms: Optional[str] = Field(None, alias="Terms")
    Days_Past_Due: Optional[int] = Field(None, alias="Days Past Due")

    # Many more columns exist; allow them transparently
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    @field_validator("Amt_Applied_to_Customer")
    @classmethod
    def _decimal_or_none(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        if v is None:
            return None
        return Decimal(str(v)).quantize(Decimal("0.01"))


class AccountOverviewRow(BaseModel):
    # Valid identifiers with JSON aliases matching desired output
    label: str
    current: Decimal = Field(..., alias="Current")
    d1_30: Decimal = Field(..., alias="1-30")
    d31_60: Decimal = Field(..., alias="31-60")
    d61_90: Decimal = Field(..., alias="61-90")
    d91_180: Decimal = Field(..., alias="91-180")
    d181p: Decimal = Field(..., alias="181+")
    total: Decimal = Field(..., alias="Total")

    model_config = ConfigDict(populate_by_name=True)


class LastPayment(BaseModel):
    date: Optional[date]
    amount: Optional[Decimal]


class MetaSummary(BaseModel):
    as_of_date: Optional[date]
    invoices_paid: Dict[str, int]          # {"L3M": 0, "LTM": 0, "since_2006": 0}
    amount_paid: Dict[str, Decimal]        # {"L3M": 0.00, "LTM": 0.00, "since_2006": 0.00}
    avg_dpd: Dict[str, Optional[float]]    # {"L3M": 0.0, "LTM": 0.0, "since_2006": 0.0}
    last_payment: LastPayment              # {"date": YYYY-MM-DD, "amount": 123.45}
    net_terms: Optional[str]
    total_credits: Decimal


class AccountOverviewRequestJSON(BaseModel):
    # JSON payloads can mirror Excel exactly; we only require these two lists
    item_list: List[ItemListRow]
    payment_history: List[PaymentHistoryRow]

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class AccountOverviewResponse(BaseModel):
    table: Dict[str, Any]  # {"columns": [...], "rows": [...]}
    meta: MetaSummary
