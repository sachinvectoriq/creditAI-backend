# core/financial_api.py
"""
Core business logic for fetching and processing financial statements from NASDAQ API.
Based on the original FinancialStatement_API.py with async improvements.
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

# ---------- Local, safe configuration defaults ----------
# ✅ CHANGE: Provide local settings + headers, so we don't depend on a
#    separate shared.config module. You can still override via env vars.
class _Settings:
    nasdaq_api_base_url: str = os.getenv(
        "NASDAQ_API_BASE_URL", "https://api.nasdaq.com"
    )  # placeholder, replace with your real internal gateway if any
    request_timeout: int = int(os.getenv("HTTP_TIMEOUT_SEC", "15"))


def get_settings() -> _Settings:
    return _Settings()


# Nasdaq endpoints often need a desktop User-Agent + Referer
HEADERS: Dict[str, str] = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": os.getenv(
        "USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119.0.0.0 Safari/537.36",
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}
# -------------------------------------------------------


settings = get_settings()


class FinancialDataFetcher:
    """Async financial data fetcher for NASDAQ API"""

    def __init__(self, ticker: str):
        """
        Initialize fetcher with ticker symbol.
        """
        self.ticker = ticker.upper()
        self.base_url = settings.nasdaq_api_base_url
        self.headers = dict(HEADERS)  # ✅ defensive copy
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry"""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session:
            await self._session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def fetch_financial_data(self, frequency: int = 2) -> Optional[Dict]:
        """
        Fetch financial data from NASDAQ API.

        Args:
            frequency: 1 for annual, 2 for quarterly

        Returns:
            JSON response data or None if request fails
        """
        # ✅ CHANGE: realistic placeholder route. Replace with the correct
        #    internal/proxy endpoint if you have one available.
        url = f"{self.base_url}/api/company/{self.ticker}/financials?frequency={frequency}"

        try:
            session = await self._get_session()
            timeout = aiohttp.ClientTimeout(total=settings.request_timeout)
            async with session.get(
                url, headers=self.headers, timeout=timeout
            ) as response:
                if response.status == 200:
                    # ✅ Some providers return text/plain; try json() with fallback
                    try:
                        return await response.json()
                    except Exception:
                        txt = await response.text()
                        logger.error("Non-JSON response received")
                        logger.debug(txt[:500])
                        return None
                else:
                    logger.error(
                        f"Request failed with status code: {response.status} | url={url}"
                    )
                    return None
        except Exception as e:
            logger.error(f"Error fetching data for {self.ticker}: {e}", exc_info=True)
            return None

    @staticmethod
    def _clean_numeric_value(value: Any) -> float:
        """
        Clean financial values by removing formatting and converting to numeric.
        Returns float; NaN for missing/invalid.
        """
        if pd.isna(value) or value == "" or value == "N/A":
            return np.nan

        value_str = str(value)

        # Handle parentheses for negative numbers
        if "(" in value_str and ")" in value_str:
            value_str = "-" + value_str.replace("(", "").replace(")", "")

        # ✅ KEEP: removing $, commas, % to enable float conversion
        value_str = value_str.replace("$", "").replace(",", "").replace("%", "")

        try:
            return float(value_str)
        except Exception:
            return np.nan

    @staticmethod
    def _format_currency(value: Union[float, int]) -> str:
        """Format numeric value with $ and commas."""
        if pd.isna(value):
            return "N/A"
        if value < 0:
            return f"$({abs(value):,.0f})"
        else:
            return f"${value:,.0f}"

    @staticmethod
    def _calculate_percentage_change(current: float, previous: float) -> float:
        """Calculate percentage change between two values."""
        if pd.isna(current) or pd.isna(previous) or previous == 0:
            return np.nan
        return round((current - previous) / abs(previous), 2)

    @staticmethod
    def _ordered_header_values(headers_dict: Dict) -> List[str]:
        """Convert headers dict into an ordered list of header values."""
        if not isinstance(headers_dict, dict):
            return []
        try:
            items = sorted(
                headers_dict.items(), key=lambda x: int(x[0].replace("value", ""))
            )
        except Exception:
            items = list(headers_dict.items())
        return [v for _, v in items]


    def _process_statement_data(
    self,
    quarterly_data: Optional[Dict],
    annual_data: Optional[Dict],
    metrics_list: List[str],
    statement_name: str,
) -> pd.DataFrame:
        """
        Process financial statement data into a DataFrame matching the output of _process_statement_data,
        without assuming a 'data' key in the payload.

        Output columns (per row dict; insertion order is preserved by pandas):
        - Metric
        - Type: 'Quarterly' or 'Annual'
        - For Quarterly:
            Q1, Q%_1, Q2, Q%_2, Q3, ... (up to Q5 if available)
        where Q%_i = % change from Q(i) -> Q(i+1)
        - For Annual:
            Y1, Y%_1, Y2, Y%_2, Y3, ... (up to Y5 if available)
        where Y%_i = % change from Y(i) -> Y(i+1)
        - Statement
        """

        # Decide which nested key we prefer based on statement_name
        preferred_key = None
        name_lower = (statement_name or "").lower()
        if "balance" in name_lower:
            preferred_key = "balanceSheetTable"
        elif "income" in name_lower:
            preferred_key = "incomeStatementTable"
        elif "cash" in name_lower:
            preferred_key = "cashFlowTable"

        def _find_rows(payload: Optional[Dict]) -> List[Dict[str, Any]]:
            if not payload or not isinstance(payload, dict):
                return []

            # 1) Prefer the statement-specific table if present anywhere
            def _find_table_by_key(d: Dict, key: str) -> Optional[Dict]:
                if key in d and isinstance(d[key], dict) and isinstance(d[key].get("rows"), list):
                    return d[key]
                for v in d.values():
                    if isinstance(v, dict):
                        hit = _find_table_by_key(v, key)
                        if hit:
                            return hit
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                hit = _find_table_by_key(item, key)
                                if hit:
                                    return hit
                return None

            if preferred_key:
                tbl = _find_table_by_key(payload, preferred_key)
                if tbl and isinstance(tbl.get("rows"), list):
                    return tbl["rows"]

            # 2) Otherwise, find the first { rows: [...] } anywhere (no 'data' assumption)
            if isinstance(payload.get("rows"), list):
                return payload["rows"]

            for v in payload.values():
                if isinstance(v, dict) and isinstance(v.get("rows"), list):
                    return v["rows"]
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict) and isinstance(item.get("rows"), list):
                            return item["rows"]

            # light recursion
            for v in payload.values():
                if isinstance(v, dict):
                    rows = _find_rows(v)
                    if rows:
                        return rows
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            rows = _find_rows(item)
                            if rows:
                                return rows
            return []

        all_data: List[Dict[str, Any]] = []

        q_rows = _find_rows(quarterly_data)
        a_rows = _find_rows(annual_data)

        # -----------------------------
        # Quarterly -> Q1..Q5 + Q%_i
        # -----------------------------
        if q_rows:
            for row in q_rows:
                metric = (row.get("value1") or "").strip()
                if metric not in metrics_list:
                    continue

                # Extract numeric values from value2..value6
                numeric_values: List[float] = []
                for i in range(2, 7):  # value2..value6
                    v = row.get(f"value{i}")
                    if v is not None:
                        numeric_values.append(self._clean_numeric_value(v))

                if not numeric_values:
                    continue

                # Build interleaved structure: Q1, Q%_1, Q2, Q%_2, ...
                row_data: Dict[str, Any] = {"Metric": metric, "Type": "Quarterly"}
                num_periods = len(numeric_values)

                for idx, val in enumerate(numeric_values):
                    # Period label
                    q_label = f"Q{idx + 1}"
                    row_data[q_label] = val

                    # Percentage-change label (between current and next), if next exists
                    if idx < num_periods - 1:
                        next_val = numeric_values[idx + 1]
                        pct_change = self._calculate_percentage_change(next_val, val)
                        # Use distinct column names for each step to avoid duplicate keys
                        pct_label = f"Q%_{idx + 1}"
                        row_data[pct_label] = pct_change

                all_data.append(row_data)

        # -----------------------------
        # Annual -> Y1..Y5 + Y%_i
        # -----------------------------
        if a_rows:
            for row in a_rows:
                metric = (row.get("value1") or "").strip()
                if metric not in metrics_list:
                    continue

                # Extract numeric values from value2..value6
                numeric_values: List[float] = []
                for i in range(2, 7):  # value2..value6
                    v = row.get(f"value{i}")
                    if v is not None:
                        numeric_values.append(self._clean_numeric_value(v))

                if not numeric_values:
                    continue

                # Build interleaved structure: Y1, Y%_1, Y2, Y%_2, ...
                row_data: Dict[str, Any] = {"Metric": metric, "Type": "Annual"}
                num_periods = len(numeric_values)

                for idx, val in enumerate(numeric_values):
                    # Period label
                    y_label = f"Y{idx + 1}"
                    row_data[y_label] = val

                    # Percentage-change label (between current and next), if next exists
                    if idx < num_periods - 1:
                        next_val = numeric_values[idx + 1]
                        pct_change = self._calculate_percentage_change(next_val, val)
                        # Use distinct column names for each step
                        pct_label = f"Y%_{idx + 1}"
                        row_data[pct_label] = pct_change

                all_data.append(row_data)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df["Statement"] = statement_name
        return df


    async def fetch_income_statement_async(self, frequency: int = 2) -> Optional[pd.DataFrame]:
        """
        Fetch and process income statement asynchronously.

        Args:
            frequency: 1 for annual, 2 for quarterly
        """
        data = await self.fetch_financial_data(frequency)
        if not data:
            return None

        # Income statement metrics
        income_metrics = [
            'Total Revenue',
            'Cost of Revenue',
            'Gross Profit',
            'Research and Development', 
            'Sales, General and Admin.',
            'Non-Recurring Items',
            'Other Operating Items',
            'Operating Expenses',
            'Earnings Before Interest and Tax',
            'Interest Expense',
            'Income Tax',
            'Net Income'
        ]

        q_task = self.fetch_financial_data(frequency=2)
        a_task = self.fetch_financial_data(frequency=1)
        quarterly_data, annual_data = await asyncio.gather(q_task, a_task, return_exceptions=False)

        return self._process_statement_data(
            quarterly_data=quarterly_data,
            annual_data=annual_data,
            metrics_list=income_metrics,
            statement_name="Income Statement",
        )

    async def fetch_balance_sheet_async(self, frequency: int = 2) -> Optional[pd.DataFrame]:
        balance_metrics = [
            'Cash and Cash Equivalents',
            'Short-Term Investments',
            'Net Receivables',
            'Inventory',
            'Total Current Assets',
            'Total Assets',
            'Working Capital',
            'Short-Term Debt / Current Portion of Long-Term Debt',
            'Accounts Payable',
            'Other Current Liabilities',
            'Total Current Liabilities',
            'Long-Term Debt',
            'Total Liabilities',
            'Net Worth(OE)',
            'Current Ratio',
            'Quick Ratio',
            'Debt/Equity'
        ]

        # Fetch both; do not “probe + early return”
        q_task = self.fetch_financial_data(frequency=2)  # Quarterly
        a_task = self.fetch_financial_data(frequency=1)  # Annual
        quarterly_data, annual_data = await asyncio.gather(q_task, a_task, return_exceptions=False)

        df = self._process_statement_data(
            quarterly_data=quarterly_data,
            annual_data=annual_data,
            metrics_list=balance_metrics,
            statement_name="Balance Sheet",
        )
        if df is None or df.empty:
            return None

        # --- Compute derived ratios: Current Ratio, Quick Ratio, Debt/Equity ---

        # Drop any existing ratio rows (in case API already returns them)
        ratio_metrics = {"Working Capital", "Net Worth(OE)", "Current Ratio", "Quick Ratio", "Debt/Equity"}
        df = df[~df["Metric"].isin(ratio_metrics)]

        # Helper: build rows for a given type (Quarterly / Annual)
        def _compute_ratio_rows(
            df_block: pd.DataFrame,
            type_label: str,
            col_prefix: str,
            statement_name: str = "Balance Sheet",
        ) -> List[Dict[str, Any]]:
            """
            df_block: df filtered to Type == type_label, still with columns like Q1..Q5 or Y1..Y5
            col_prefix: 'Q' for quarterly, 'Y' for annual
            """
            if df_block.empty:
                return []

            # Work with metric as index
            block = df_block.set_index("Metric")
            cols = [c for c in block.columns if c.startswith(col_prefix)]
            rows: List[Dict[str, Any]] = []

            # Safely check required metrics
            idx = block.index

            def _row_data(metric_name: str, values: pd.Series) -> Dict[str, Any]:
                out: Dict[str, Any] = {
                    "Metric": metric_name,
                    "Type": type_label,
                    "Statement": statement_name,
                }
                for c in cols:
                    out[c] = values.get(c, np.nan)
                return out
            
            #Working Capital = Current Assets - Current Liabilities
            if {"Total Current Assets", "Total Current Liabilities"}.issubset(idx):
                ca = block.loc["Total Current Assets", cols]
                cl = block.loc["Total Current Liabilities", cols]
                working_capital = ca - cl
                rows.append(_row_data("Working Capital", working_capital))

            # Net Worth(OE) = Total Assets - Total Liabilities
            if {"Total Assets", "Total Liabilities"}.issubset(idx):
                ta = block.loc["Total Assets", cols]
                tl = block.loc["Total Liabilities", cols]
                net_worth = ta - tl
                rows.append(_row_data("Net Worth(OE)", net_worth))

            # Current Ratio = Current Assets / Current Liabilities
            if {"Total Current Assets", "Total Current Liabilities"}.issubset(idx):
                ca = block.loc["Total Current Assets", cols]
                cl = block.loc["Total Current Liabilities", cols]
                cl = cl.replace(0, np.nan)
                with np.errstate(divide="ignore", invalid="ignore"):
                    current_ratio = ca / cl
                rows.append(_row_data("Current Ratio", current_ratio))

            # Quick Ratio = (Current Assets - Inventory) / Current Liabilities
            if {
                "Total Current Assets",
                "Inventory",
                "Total Current Liabilities",
            }.issubset(idx):
                ca = block.loc["Total Current Assets", cols]
                inv = block.loc["Inventory", cols]
                cl = block.loc["Total Current Liabilities", cols]
                cl = cl.replace(0, np.nan)
                with np.errstate(divide="ignore", invalid="ignore"):
                    quick_ratio = (ca - inv) / cl
                rows.append(_row_data("Quick Ratio", quick_ratio))

            # Debt/Equity = Total Liabilities / Net Worth (OE)
            if {"Total Liabilities", "Net Worth(OE)"}.issubset(idx):
                tl = block.loc["Total Liabilities", cols]
                nw = block.loc["Net Worth(OE)", cols]
                nw = nw.replace(0, np.nan)
                with np.errstate(divide="ignore", invalid="ignore"):
                    debt_equity = tl / nw
                rows.append(_row_data("Debt/Equity", debt_equity))

            return rows

        # Split quarterly / annual
        df_quarterly = df[df["Type"] == "Quarterly"]
        df_annual = df[df["Type"] == "Annual"]

        new_rows: List[Dict[str, Any]] = []
        new_rows.extend(_compute_ratio_rows(df_quarterly, "Quarterly", "Q"))
        new_rows.extend(_compute_ratio_rows(df_annual, "Annual", "Y"))

        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

        # Optionally re-order metrics to follow balance_metrics order
        # (keeps your ratios at the end or in the same logical order)
        metric_order = {m: i for i, m in enumerate(balance_metrics)}
        df["Metric_order"] = df["Metric"].map(metric_order).fillna(len(balance_metrics))
        df = df.sort_values(["Type", "Metric_order"]).drop(columns=["Metric_order"])

        return df if not df.empty else None

    async def fetch_cash_flow_async(self, frequency: int = 2) -> Optional[pd.DataFrame]:
        """
        Fetch and process cash flow statement asynchronously.

        Args:
            frequency: 1 for annual, 2 for quarterly
        """

        cashflow_metrics = [
            'Net Income',
            'Cash Flows-Operating Activities',
            'Depreciation',
            'Net Income Adjustments',
            'Accounts Receivable',
            'Changes in Inventories',
            'Other Operating Activities',
            'Liabilities',
            'Net Cash Flow-Operating',
            'Cash Flows-Investing Activities',
            'Capital Expenditures',
            'Investments',
            'Other Investing Activities',
            'Net Cash Flows-Investing',
            'Cash Flows-Financing Activities',
            'Sale and Purchase of Stock',
            'Net Borrowings',
            'Other Financing Activities',
            'Net Cash Flows-Financing',
            'Net Cash Flow'
        ]

        q_task = self.fetch_financial_data(frequency=2)  # Quarterly
        a_task = self.fetch_financial_data(frequency=1)  # Annual
        quarterly_data, annual_data = await asyncio.gather(q_task, a_task, return_exceptions=False)

        df =  self._process_statement_data(
            quarterly_data=quarterly_data,
            annual_data=annual_data,
            metrics_list=cashflow_metrics,
            statement_name="Cash Flow",
        )
        return None if df is None or df.empty else df

    async def calculate_financial_ratios_async(self, financial_data: Dict) -> Dict[str, float]:
        """
        Calculate key financial ratios from financial statements.

        NOTE: This method expects a normalized financial_data dict. If you adapt
              the payload, adjust the key paths here accordingly.
        """
        ratios: Dict[str, float] = {}

        try:
            balance_sheet = financial_data.get("balance_sheet") or {}
            income_statement = financial_data.get("income_statement") or {}

            current_assets = balance_sheet.get("current_assets", 0)
            current_liabilities = balance_sheet.get("current_liabilities", 0)
            if current_liabilities:
                ratios["current_ratio"] = round(current_assets / current_liabilities, 2)

            total_debt = balance_sheet.get("total_debt", 0)
            total_equity = balance_sheet.get("total_equity", 0)
            if total_equity:
                ratios["debt_to_equity"] = round(total_debt / total_equity, 2)

            net_income = income_statement.get("net_income", 0)
            total_assets = balance_sheet.get("total_assets", 0)
            if total_assets:
                ratios["return_on_assets"] = round((net_income / total_assets) * 100, 2)

            if total_equity:
                ratios["return_on_equity"] = round((net_income / total_equity) * 100, 2)

            revenue = income_statement.get("total_revenue", 0)
            gross_profit = income_statement.get("gross_profit", 0)
            if revenue:
                ratios["gross_margin"] = round((gross_profit / revenue) * 100, 2)

            operating_income = income_statement.get("operating_income", 0)
            if revenue:
                ratios["operating_margin"] = round((operating_income / revenue) * 100, 2)

            if revenue:
                ratios["net_margin"] = round((net_income / revenue) * 100, 2)

        except Exception as e:
            logger.error(f"Error calculating financial ratios: {e}", exc_info=True)

        return ratios

    async def fetch_all_statements_async(self, frequency: int = 2) -> Optional[Dict]:
        """
        Fetch all financial statements in parallel.

        Args:
            frequency: 1 for annual, 2 for quarterly
        """
        try:
            tasks = [
                self.fetch_income_statement_async(frequency),
                self.fetch_balance_sheet_async(frequency),
                self.fetch_cash_flow_async(frequency),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            income_statement, balance_sheet, cash_flow = (None, None, None)
            names = ["income", "balance", "cashflow"]
            out = []

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching {names[i]} statement: {result}")
                    out.append(None)
                else:
                    out.append(result)

            income_statement, balance_sheet, cash_flow = out

            # ✅ CHANGE: Convert DataFrames to dict records safely
            return {
                "ticker": self.ticker,
                "frequency": "quarterly" if frequency == 2 else "annual",
                "income_statement": (
                    income_statement.to_dict("records")
                    if (income_statement is not None and not income_statement.empty)
                    else None
                ),
                "balance_sheet": (
                    balance_sheet.to_dict("records")
                    if (balance_sheet is not None and not balance_sheet.empty)
                    else None
                ),
                "cash_flow": (
                    cash_flow.to_dict("records")
                    if (cash_flow is not None and not cash_flow.empty)
                    else None
                ),
                "metadata": {
                    "last_updated": datetime.utcnow().isoformat(),
                    "source": "NASDAQ API",
                },
            }
        except Exception as e:
            logger.error(f"Error fetching all statements: {e}", exc_info=True)
            return None

    async def close(self):
        """Close the aiohttp session"""
        if self._session:
            await self._session.close()
            self._session = None


# Utility function for creating fetcher with context manager
async def fetch_financial_data_for_ticker(ticker: str, frequency: int = 2) -> Optional[Dict]:
    """
    Convenience function to fetch financial data for a ticker.
    """
    async with FinancialDataFetcher(ticker) as fetcher:
        return await fetcher.fetch_all_statements_async(frequency)
