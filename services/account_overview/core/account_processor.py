# account_processor.py — core compute/transform functions (pure)
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from services.account_overview.model import ItemListRow, PaymentHistoryRow, AgingBucket

# ---- Output labels (kept from your existing intent) ----
ROW_LABELS: List[str] = [
    "OSC01",
    "Aerotek",
    "OCS03",
    "Aviation",
    "Aston Carter",
    "SJA01",
    "Actalent",
    "CE",
    "Scientific",
    "Services",
    "Actalent Canada",
    "Services_EASCA",
    "Aerotek Canada",
    "Aston Carter Canada",
    "MLA/IEL",
    "Teksystems",
    "Tek Global",
    "Totals",
]

# Map Excel Unit -> output row label (keep/extend as needed)
UNIT_TO_ROW: Dict[str, str] = {
    "OCS01": "OSC01",
    "OCS03": "OCS03",
    "OAV01": "Aviation",
    "SJA01": "SJA01",
    "OCS02": "CE",
    "ASC01": "Scientific",
    "INP01": "Services",
    "CACOR": "Actalent Canada",
    "EASCA": "Services_EASCA",
    "CAIND": "Aerotek Canada",
    "CAAC1": "Aston Carter Canada",
    "IELO1": "MLA/IEL",
    "TEK01": "Teksystems",
    "TKC01": "Tek Global",
    # If some units should directly display by name, add here
}

AGING_COLS = [
    AgingBucket.CURRENT.value,
    AgingBucket.D1_30.value,
    AgingBucket.D31_60.value,
    AgingBucket.D61_90.value,
    AgingBucket.D91_180.value,
    AgingBucket.D181P.value,
]
ALL_COLS = AGING_COLS + [AgingBucket.TOTAL.value]


def _bucket_for_days(days_late: int) -> str:
    if days_late <= 0:
        return AgingBucket.CURRENT.value
    if 0 < days_late <= 30:
        return AgingBucket.D1_30.value
    if 30 < days_late <= 60:
        return AgingBucket.D31_60.value
    if 60 < days_late <= 90:
        return AgingBucket.D61_90.value
    if 90 < days_late <= 180:
        return AgingBucket.D91_180.value
    return AgingBucket.D181P.value


def _d2(x: Optional[float | int | Decimal]) -> Optional[Decimal]:
    if x is None:
        return None
    return Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# ---------- API: load_inputs ----------
def load_inputs(
    item_rows: Iterable[ItemListRow] | pd.DataFrame,
    payment_rows: Iterable[PaymentHistoryRow] | pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Accepts validated rows (from JSON) or raw DataFrames (from Excel) and returns clean DataFrames.
    Columns are aligned to: Unit, Days_Late, Item_Balance | Payment_Date, Amt_Applied_to_Customer, Terms, Days_Past_Due
    """
    if isinstance(item_rows, pd.DataFrame):
        items_df = item_rows.rename(
            columns={"Days Late": "Days_Late", "Item Balance": "Item_Balance"}
        ).copy()
    else:
        # JSON already validated with aliases matching Excel headers
        items_df = pd.DataFrame([r.model_dump(by_alias=True) for r in item_rows]).rename(
            columns={"Days Late": "Days_Late", "Item Balance": "Item_Balance"}
        )

    if isinstance(payment_rows, pd.DataFrame):
        pay_df = payment_rows.rename(
            columns={
                "Payment Date": "Payment_Date",
                "Amt Applied to Customer": "Amt_Applied_to_Customer",
                "Days Past Due": "Days_Past_Due",
            }
        ).copy()
    else:
        pay_df = pd.DataFrame([r.model_dump(by_alias=True) for r in payment_rows]).rename(
            columns={
                "Payment Date": "Payment_Date",
                "Amt Applied to Customer": "Amt_Applied_to_Customer",
                "Days Past Due": "Days_Past_Due",
            }
        )

    # Coercions
    items_df["Unit"] = items_df["Unit"].astype(str).str.strip()
    items_df["Days_Late"] = pd.to_numeric(items_df["Days_Late"], errors="coerce").fillna(0).astype(int)
    items_df["Item_Balance"] = pd.to_numeric(items_df["Item_Balance"], errors="coerce").fillna(0.0)

    pay_df["Payment_Date"] = pd.to_datetime(pay_df["Payment_Date"], errors="coerce").dt.date
    pay_df["Amt_Applied_to_Customer"] = pd.to_numeric(pay_df["Amt_Applied_to_Customer"], errors="coerce")
    pay_df["Days_Past_Due"] = pd.to_numeric(pay_df["Days_Past_Due"], errors="coerce")

    return items_df, pay_df


# ---------- API: build_account_overview ----------
def build_account_overview(items_df: pd.DataFrame, pay_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Make aging table + meta KPIs using pure, deterministic transforms.
    """
    # Initialize result table
    table = pd.DataFrame(0.0, index=ROW_LABELS, columns=ALL_COLS, dtype="float64")

    # Aggregate by mapped rows and aging buckets
    for _, row in items_df.iterrows():
        unit = row.get("Unit")
        days = int(row.get("Days_Late", 0))
        bal = float(row.get("Item_Balance", 0.0))
        target = UNIT_TO_ROW.get(unit)
        if not target:
            continue
        table.loc[target, _bucket_for_days(days)] += bal

    # Row totals
    table[AgingBucket.TOTAL.value] = table[AGING_COLS].sum(axis=1)

    # Totals row aggregation
    mapped_labels = sorted(set(UNIT_TO_ROW.values()))
    if mapped_labels:
        table.loc["Totals", AGING_COLS] = table.loc[mapped_labels, AGING_COLS].sum(axis=0)
        table.loc["Totals", AgingBucket.TOTAL.value] = table.loc["Totals", AGING_COLS].sum()

    # Meta KPIs
    as_of: Optional[date] = None
    if not pay_df.empty and pay_df["Payment_Date"].notna().any():
        as_of = max(d for d in pay_df["Payment_Date"] if pd.notna(d))
    if as_of is None:
        as_of = date(2024, 9, 8)  # deterministic fallback

    cutoff_90 = as_of - timedelta(days=91)
    cutoff_365 = as_of - timedelta(days=365)

    l3m_mask = pay_df["Payment_Date"] >= cutoff_90
    ltm_mask = pay_df["Payment_Date"] >= cutoff_365

    l3m_paid_cnt = int(pay_df.loc[l3m_mask, "Payment_Date"].notna().sum())
    ltm_paid_cnt = int(pay_df.loc[ltm_mask, "Payment_Date"].notna().sum())
    since_cnt = int(pay_df["Amt_Applied_to_Customer"].notna().sum())

    l3m_amt = pay_df.loc[l3m_mask, "Amt_Applied_to_Customer"].sum(skipna=True)
    ltm_amt = pay_df.loc[ltm_mask, "Amt_Applied_to_Customer"].sum(skipna=True)
    since_amt = pay_df["Amt_Applied_to_Customer"].sum(skipna=True)

    l3m_dpd = pay_df.loc[l3m_mask, "Days_Past_Due"].mean()
    ltm_dpd = pay_df.loc[ltm_mask, "Days_Past_Due"].mean()
    since_dpd = pay_df["Days_Past_Due"].mean()

    last_date = None
    last_amt = None
    if not pay_df.empty and pay_df["Payment_Date"].notna().any():
        last_date = max(d for d in pay_df["Payment_Date"] if pd.notna(d))
        same_day = pay_df["Payment_Date"] == last_date
        last_amt = pay_df.loc[same_day, "Amt_Applied_to_Customer"].sum(skipna=True)

    net_terms = None
    if "Terms" in pay_df.columns:
        nt = pay_df["Terms"].dropna()
        if not nt.empty:
            net_terms = str(nt.iloc[0])

    total_credits = items_df.loc[items_df["Item_Balance"] < 0, "Item_Balance"].sum(skipna=True)

    meta = {
        "as_of_date": as_of,
        "invoices_paid": {"L3M": l3m_paid_cnt, "LTM": ltm_paid_cnt, "since_2006": since_cnt},
        "amount_paid": {
            "L3M": _d2(l3m_amt) or Decimal("0.00"),
            "LTM": _d2(ltm_amt) or Decimal("0.00"),
            "since_2006": _d2(since_amt) or Decimal("0.00"),
        },
        "avg_dpd": {
            "L3M": float(l3m_dpd) if pd.notna(l3m_dpd) else None,
            "LTM": float(ltm_dpd) if pd.notna(ltm_dpd) else None,
            "since_2006": float(since_dpd) if pd.notna(since_dpd) else None,
        },
        "last_payment": {
            "date": last_date,
            "amount": _d2(last_amt) if last_amt is not None else None,
        },
        "net_terms": net_terms,
        "total_credits": _d2(total_credits) or Decimal("0.00"),
    }

    return table, meta


# ---------- API: format_overview_table ----------
def format_overview_table(table: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert DataFrame to a JSON-table with stable ordering and fixed precision.
    """
    table = table.loc[ROW_LABELS, ALL_COLS].fillna(0.0)
    rows = []
    for label, r in table.iterrows():
        rows.append(
            {
                "label": label,
                "Current": round(float(r["Current"]), 2),
                "1-30": round(float(r["1-30"]), 2),
                "31-60": round(float(r["31-60"]), 2),
                "61-90": round(float(r["61-90"]), 2),
                "91-180": round(float(r["91-180"]), 2),
                "181+": round(float(r["181+"]), 2),
                "Total": round(float(r["Total"]), 2),
            }
        )
    return {"columns": ["label"] + ALL_COLS, "rows": rows}
