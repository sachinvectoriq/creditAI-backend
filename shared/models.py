"""
Shared Pydantic models for request and response validation across all microservices.
"""
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime, date
from enum import Enum
from decimal import Decimal



# Common Enums
class AnalysisType(str, Enum):
    """Types of AI analysis available"""
    RISK = "risk"
    LIQUIDITY = "liquidity"
    PROFITABILITY = "profitability"
    CASHFLOW = "cashflow"
    RECOMMENDATION = "recommendation"


class FrequencyType(str, Enum):
    """Financial statement frequency"""
    ANNUAL = "annual"
    QUARTERLY = "quarterly"


class FileSource(str, Enum):
    """Source of 10-Q filing"""
    SEC_AUTO = "sec_auto"
    UPLOAD = "upload"


# Base Response Models
class BaseResponse(BaseModel):
    """Base response model with common fields"""
    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseResponse):
    """Error response model"""
    success: bool = False
    error: str
    details: Optional[Dict[str, Any]] = None


# Financial Statement Models
class FinancialStatementRequest(BaseModel):
    """Request model for financial statements"""
    ticker: str = Field(..., description="Stock ticker symbol", min_length=1, max_length=10)
    frequency: FrequencyType = FrequencyType.QUARTERLY
    
    @validator('ticker')
    def ticker_uppercase(cls, v):
        return v.upper().strip()


class FinancialMetric(BaseModel):
    """Individual financial metric"""
    name: str
    value: Optional[Union[float, str]] = None
    formatted_value: Optional[str] = None
    percentage_change: Optional[float] = None
    period: Optional[str] = None


class FinancialStatement(BaseModel):
    """Financial statement data"""
    statement_type: str  # income_statement, balance_sheet, cash_flow
    ticker: str
    frequency: str
    periods: List[str]
    metrics: Dict[str, List[FinancialMetric]]
    metadata: Optional[Dict[str, Any]] = None


class FinancialStatementResponse(BaseResponse):
    """Response model for financial statements"""
    data: FinancialStatement
    cached: bool = False


# Account Overview Models
class AccountOverviewRequest(BaseModel):
    """Request model for account overview analysis"""
    # Files will be uploaded as multipart form data
    # This model is for metadata
    analysis_date: Optional[datetime] = Field(default_factory=datetime.utcnow)
    include_summary: bool = True


class AccountMetrics(BaseModel):
    """Account metrics summary"""
    current: Optional[float] = None
    days_1_30: Optional[float] = None
    days_31_60: Optional[float] = None
    days_61_90: Optional[float] = None
    days_91_180: Optional[float] = None
    days_181_plus: Optional[float] = None
    total: Optional[float] = None


class AccountSummary(BaseModel):
    """Account summary statistics"""
    l3m_invoices_paid: Optional[int] = None
    ltm_invoices_paid: Optional[int] = None
    total_invoices_paid: Optional[int] = None
    l3m_amount_paid: Optional[float] = None
    ltm_amount_paid: Optional[float] = None
    total_amount_paid: Optional[float] = None
    l3m_avg_dpd: Optional[float] = None
    ltm_avg_dpd: Optional[float] = None
    total_avg_dpd: Optional[float] = None
    last_payment_date: Optional[str] = None
    last_payment_amount: Optional[float] = None
    net_terms: Optional[str] = None
    total_credits: Optional[float] = None


class AccountOverviewResponse(BaseResponse):
    """Response model for account overview"""
    overview_table: Dict[str, AccountMetrics]
    summary: Optional[AccountSummary] = None
    raw_data: Optional[Dict[str, Any]] = None


# AI Analysis Models
class AIAnalysisRequest(BaseModel):
    """Request model for AI analysis"""
    ticker: str = Field(..., description="Stock ticker symbol")
    analysis_types: List[AnalysisType] = Field(
        default=[AnalysisType.RISK, AnalysisType.LIQUIDITY, AnalysisType.PROFITABILITY]
    )
    file_source: FileSource = FileSource.SEC_AUTO
    # If file_source is UPLOAD, file will be sent as multipart
    
    @validator('ticker')
    def ticker_uppercase(cls, v):
        return v.upper().strip()


class AIAnalysisResult(BaseModel):
    """Individual AI analysis result"""
    analysis_type: AnalysisType
    content: str
    confidence_score: Optional[float] = None
    key_findings: Optional[List[str]] = None
    data_sources: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AIAnalysisResponse(BaseResponse):
    """Response model for AI analysis"""
    ticker: str
    analyses: List[AIAnalysisResult]
    document_source: str
    processing_time: Optional[float] = None


# SEC Filing Models
class SECFilingInfo(BaseModel):
    """SEC filing information"""
    ticker: str
    cik: str
    filing_type: str = "10-Q"
    filing_date: Optional[str] = None
    accession_number: Optional[str] = None
    document_url: Optional[str] = None
    pdf_url: Optional[str] = None
    html_url: Optional[str] = None


class SECFilingResponse(BaseResponse):
    """Response model for SEC filing retrieval"""
    filing_info: SECFilingInfo
    content: Optional[str] = None
    extracted_text: Optional[str] = None


# Batch Processing Models
class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis of multiple companies"""
    tickers: List[str] = Field(..., description="List of ticker symbols", min_items=1, max_items=50)
    analysis_types: List[AnalysisType] = Field(default_factory=lambda: [AnalysisType.RISK])
    parallel: bool = True
    
    @validator('tickers')
    def tickers_uppercase(cls, v):
        return [ticker.upper().strip() for ticker in v]


class BatchAnalysisResponse(BaseResponse):
    """Response model for batch analysis"""
    results: Dict[str, Union[AIAnalysisResponse, ErrorResponse]]
    total_processed: int
    successful: int
    failed: int
    processing_time: float


# Validation helpers
def validate_ticker(ticker: str) -> str:
    """Validate and normalize ticker symbol"""
    ticker = ticker.upper().strip()
    if not ticker or len(ticker) > 10:
        raise ValueError(f"Invalid ticker symbol: {ticker}")
    if not ticker.replace('-', '').replace('.', '').isalnum():
        raise ValueError(f"Ticker contains invalid characters: {ticker}")
    return ticker


def validate_file_type(filename: str, allowed_types: List[str]) -> bool:
    """Validate file type based on extension"""
    if not filename:
        return False
    ext = filename.split('.')[-1].lower()
    return ext in allowed_types


# Report Generation Models
class ReportGenerationRequest(BaseModel):
    """Request model for comprehensive report generation"""
    ticker: str
    include_financials: bool = True
    include_ai_analysis: bool = True
    include_account_overview: bool = False
    report_format: str = "html"  # html, pdf, json
    
    @validator('ticker')
    def ticker_uppercase(cls, v):
        return v.upper().strip()


class ReportSection(BaseModel):
    """Individual section of a report"""
    title: str
    content: Union[str, Dict[str, Any]]
    section_type: str  # text, table, chart
    order: int


class ReportGenerationResponse(BaseResponse):
    """Response model for report generation"""
    ticker: str
    report_sections: List[ReportSection]
    report_url: Optional[str] = None
    download_url: Optional[str] = None
    format: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# models for account overview service
# model.py — Pydantic request/response models & enums



class AgingBucket(str, Enum):
    CURRENT = "Current"
    D1_30 = "1-30"
    D31_60 = "31-60"
    D61_90 = "61-90"
    D91_180 = "91-180"
    D181P = "181+"
    TOTAL = "Total"


class ItemListRow(BaseModel):
    Unit: str = Field(..., description="Business unit code (e.g., OCS01, OAV01)")
    Days_Late: int = Field(..., alias="Days Late", ge=-10_000, le=10_000)
    Item_Balance: Decimal = Field(..., alias="Item Balance")

    class Config:
        populate_by_name = True


class PaymentHistoryRow(BaseModel):
    Payment_Date: Optional[date] = Field(None, alias="Payment Date")
    Amt_Applied_to_Customer: Optional[Decimal] = Field(None, alias="Amt Applied to Customer")
    Terms: Optional[str] = None
    Days_Past_Due: Optional[int] = Field(None, alias="Days Past Due")

    class Config:
        populate_by_name = True


class AccountOverviewRow(BaseModel):
    label: str
    Current: Decimal
    _1_30: Decimal = Field(..., alias="1-30")
    _31_60: Decimal = Field(..., alias="31-60")
    _61_90: Decimal = Field(..., alias="61-90")
    _91_180: Decimal = Field(..., alias="91-180")
    _181p: Decimal = Field(..., alias="181+")
    Total: Decimal

    class Config:
        populate_by_name = True


class AccountOverviewRequestJSON(BaseModel):
    item_list: List[ItemListRow] = Field(..., description="Array mirroring item_list.xlsx rows")
    payment_history: List[PaymentHistoryRow] = Field(..., description="Array mirroring payment_history.xlsx rows")

    @validator("item_list")
    def non_empty_item_list(cls, v):
        if not v:
            raise ValueError("item_list cannot be empty")
        return v


class MetaSummary(BaseModel):
    as_of_date: Optional[date]
    invoices_paid: Dict[str, int]  # {"L3M": 0, "LTM": 0, "since_2006": 0}
    amount_paid: Dict[str, Decimal]  # {"L3M": 0, "LTM": 0, "since_2006": 0}
    avg_dpd: Dict[str, Optional[float]]  # {"L3M": 0.0, "LTM": 0.0, "since_2006": 0.0}
    last_payment: Dict[str, Optional[str]]  # {"date": "YYYY-MM-DD", "amount": "1234.00"}
    net_terms: Optional[str]
    total_credits: Decimal


class AccountOverviewResponse(BaseModel):
    table: Dict[str, Any]  # {"columns": [...], "rows": [...]}
    meta: MetaSummary
