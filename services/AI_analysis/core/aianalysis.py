import os
import io
import re
import json
import asyncio
import logging
from typing import Optional, Dict, Any

import requests
import pandas as pd
from bs4 import BeautifulSoup
import PyPDF2
import docx2txt

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor

from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv

logger = logging.getLogger("ai-analysis-core")

# Load .env
load_dotenv()

# ------------------------------------------------------------
# 🔍 Reflection Prompt (unchanged)
# ------------------------------------------------------------
REFLECTION_PROMPT = (
    "1) Verify whether the summary is accurately derived from its prompt without hallucination.\n"
    "2) Ensure all numeric details are correct, consistent, and neatly represented.\n"
    "3) Highlight any factual or stylistic issues clearly."
)

# ------------------------------------------------------------
# ⚙️ Generator Agent Configurations (UNCHANGED prompts)
# ------------------------------------------------------------
GENERATOR_CONFIGS = [
    {
        "id": "asst_b91VJXcKj2vdEyJrzqmEQxoe",
        "name": "risk_analyst",
        "rag_prompt": """Debt maturity- From the given 10-Q filing, provide a 2-3 sentence narrative covering debt maturity including total debt, near-term maturities, long-term maturities, interest rates, principal amount, carrying amount, and repayment or refinancing details. Include specific dollar amounts and dates where available. Write as continuous text, not bullet points.
            Interest Expense- Analyze the 10-Q filing and provide a 2-3 sentence summary about the company's debt interest expenses or interest rates. Include Net Interest expenses of that quarter and same quarter of previous year if available. Include specific dollar amounts and percentages. If not available, respond exactly: "This details are not provided in 10-Q filing". Write as continuous text, not bullet points.
            Executive Summary- Analyse how financial risk factors could negatively impact operations and cashflow presented in the latest 10-Q filing. Provide a 3-4 sentence narrative summary identifying how financial/debt/operational risks could negatively impact operations and cashflow. Include specific figures where applicable (e.g., $4,900 thousand or $4.9 million).""",
        "base_prompt": """You are a financial analyst. Based on the  risk analysis details, generate a concise summary of 160 words that captures all key insights and data points without losing important context:
            Ensure the summary:

                Covers all major profitability metrics and trends (e.g., Financial and Debt-Related Risks, Debt maturity, Interest Expense etc.).

                Highlights significant changes or drivers affecting profitability.

                Remains factual, precise, and compact—no filler sentences or loss of information.
                Use specific figures and dates where available.

                Output only the final summarized paragraph, not bullet points or commentary.
                In starting the summary, include the currency scale used in the report (e.g., "in millions", "in thousands")—typically found near the balance sheet or income statement headings.

Context:
{rag_context}
""",
        "revise_prompt": """Improve the financial risk analysis summary based on reflection feedback.
Ensure factual correctness, consistent terminology, and clarity. Keep around 160 words with crisp financial language."""
    },
    {
        "id": "asst_el50EEqzYWyo6AXvYUA4QuFD",
        "name": "profitability",
        "rag_prompt": """Analyze the company’s profit growth based on the provided 10-Q filing. Write in a concise, analytical tone similar to an investor earnings summary—no bullet points, only a clear narrative using concrete financial figures and trends (e.g., “$ millions,” “x coverage”).

            In 2–3 paragraphs, summarize:

            Revenue performance: total revenue vs. prior quarter and year-ago quarter, key growth or decline trends.

            Operating income and margins: discuss operating income, gross/operating/net profit margins, and major cost drivers affecting profitability.

            Expense drivers and growth outlook: highlight notable cost changes, year-over-year profit growth, and any forward-looking guidance or risks mentioned in the filing.""",
        "base_prompt": """You are a financial analyst. Using the provided Profitability Analysis, write a single paragraph (~150 words) that preserves all key facts. Cover revenue, operating income, gross/operating/net margins, expenses, year-over-year and quarter-over-quarter growth, and any forward-looking statements. 
        Highlight significant changes and their drivers. Be factual, precise, and compact—no filler. Use specific figures and dates wherever available. Output only the final paragraph (no bullets, headings, or extra commentary).

Context:
{rag_context}
""",
        "revise_prompt": """Refine the profitability summary based on reflection feedback.
Maintain clear, concise, and accurate financial metrics description."""
    },
    {
        "id": "asst_KCnEF6s5cXRypdsy8YeHG6UJ",
        "name": "cash_flow",
        "rag_prompt": """From the given 10-Q filing of a company, focusing specifically on the *Cash Flow* section:

        * Cash flow from operating activities
        * Cash flow from investing activities
        * Cash flow from financing activities

        Include the starting amount, ending amount, comparison to the same period in the prior year, and explain the key reasons behind the changes.""",
        "base_prompt": """You are a cash flow specialist. Analyze liquidity, debt structure, and interest cost impacts on financial flexibility from the context below , summarize the changes in the following cash flow activities in 3–4 sentences.

Context:
{rag_context}
""",
        "revise_prompt": """Enhance the cash flow summary using reflection feedback.
Focus on liquidity dynamics and debt repayment schedules with precision and clarity."""
    },
    {
        "id": "asst_MGM8MGigFossiOxAX4m5U4qy",
        "name": "liquidity",
        "rag_prompt": """"From the company’s 10-Q, report the following in plain text using exact figures/units (millions/billions) and the exact quarter-end date. Keep to 4 short labeled sections; no extra commentary.

Cash & Cash Equivalents: State the precise amount and the quarter-end date as reported.

Liquidity Runway (12 months): Say whether management indicates resources will last 12 months; include the key line or clear paraphrase from management’s disclosure.

Credit Facilities: Provide total facility size, amount drawn/outstanding, undrawn availability, maturity/expiration date, and any notable Credit Agreement details.

Going Concern: State if the filing discloses substantial doubt about continuing as a going concern; if none, say so explicitly.""",
        "base_prompt": """You are a financial analyst. Using the provided Liquidity Analysis, write a single paragraph (~150 words) that preserves all key details. Cover: (1) Cash & cash equivalents (amount and quarter-end date), (2) 12-month liquidity outlook (management’s view on runway), (3) line of credit details (total, drawn/undrawn, maturity/expiration, key credit agreement terms), (4) going concern status. 
        Highlight notable changes or drivers affecting liquidity. Be factual, precise, and compact—no filler. Use specific figures and dates wherever available. Output only the final paragraph (no bullets, headings, or extra commentary).
Context:
{rag_context}
""",
        "revise_prompt": """Improve the market risk summary based on reflection feedback.
Highlight impacts of downgrades, rate volatility, and refinancing conditions clearly."""
    },
    {
        "id": "asst_ISF54ANWtFB3KsyVLpdcu41R",
        "name": "executive_summary",
        "rag_prompt": """ You are a financial-extraction and credit-analysis agent. Perform a two-stage pipeline, but return only the final Step-3 output.

Stage A — Extraction (from 10-Q/10-K/Nasdaq text):

Extract the most recent quarter’s figures into an internal JSON with fields: Company, Report_Date; Liquidity (Cash_and_Equivalents, Total_Current_Assets, Total_Current_Liabilities, Current_Ratio = Assets/Liabilities, Operating_Cash_Flow, Liquidity_Runway_Months = if OCF<0 then Cash / |OCF/12| else “Not applicable”); Leverage (Total_Debt, Shareholders_Equity, Debt_to_Equity = Debt/Equity, Debt_Maturities {2025…2029_and_beyond}, Undrawn_Facilities); Profitability (Revenue, Operating_Income, Net_Income, Operating_Margin = OpInc/Revenue); Cash_Flow (Operating_Cash_Flow, Capex, Free_Cash_Flow = OCF–Capex); Commitments_Contingencies (Purchase_Obligations, Legal_Tax_Exposure).

Rules: use exact reported values and currency units; do not invent values—leave missing as empty; compute ratios only if both inputs exist; find debt maturities in notes; find undrawn facilities in credit agreement/liquidity sections.

Stage B — Rating Summary (from the extracted JSON only):

Build a one-page “System Preliminary Credit Rating Summary” with sections: Company Snapshot; Liquidity & Cash Flow; Debt & Capital Structure; Profitability; Commitments & Contingencies; Peer Benchmark (“Not disclosed” if missing); Risk Flags (🔴/🟠/🟢); System Preliminary Rating Guidance (Risk Level, Equivalent Rating Band, Suggested Action, and the disclaimer: “This is system preliminary guidance only. Final decision rests with the Credit Compliance Team.”).

Rules: max 2 sentences commentary per section; do not invent missing values; ratios must match extracted JSON.

Final Required Output (return only this, no extra text):

Commentary Summary: 3–5 concise bullet points covering the key findings across sections.

Risk Flags: list exactly as determined (🔴/🟠/🟢).

System Preliminary Rating Guidance: show the final guidance exactly.""",
        "base_prompt": """Source document: Attached 10-Q/10-K/Nasdaq filing text.
Please extract the latest quarter’s data per the schema (including debt maturities from notes and undrawn facilities from credit/liquidity sections), generate the one-page rating summary from the extracted values only, and then return only:

Commentary Summary:

Point 1

Point 2

Point 3

Point 4

Risk Flags:
[🔴/🟠/🟢 items]

System Preliminary Rating Guidance:

Risk Level: [Low/Medium/High]

Equivalent Rating Band: [AAA–BBB / BB–B / CCC–D]

Suggested Action: [Increase / Maintain / Reduce]

Disclaimer: “This is system preliminary guidance only. Final decision rests with the Credit Compliance Team.”

Context:
{rag_context}
""",
        "revise_prompt": """Refine the executive summary based on reflection feedback.
Make it more cohesive, readable, and precise, preserving all numeric facts."""
    }
]

# ------------------------------------------------------------
# 🧠 Financial RAG Agentic System (async) — reorganized only
# ------------------------------------------------------------
class FinancialRAGMAgenticSystem:
    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        embedding_deployment: str,
        openai_deployment: str,
        api_version: str = "2024-02-01",
    ) -> None:
        self.embed_model = AzureOpenAIEmbedding(
            model=embedding_deployment,
            deployment_name=embedding_deployment,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        self.llm = AzureOpenAI(
            model=openai_deployment,
            deployment_name=openai_deployment,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            temperature=0.1,
            max_tokens=4000,
        )
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        self.vector_index = None
        self.source_text = ""
        self.executor = ThreadPoolExecutor(max_workers=10)

    async def _fetch_and_parse_10q(self, url: str) -> str:
        def blocking_fetch() -> str:
            headers = {
                "User-Agent": os.getenv("SEC_USER_AGENT", "creditai@example.com"),
                "Accept-Encoding": "gzip, deflate",
                "Accept-Language": "en-US,en;q=0.9",
            }
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            return soup.get_text(separator="\n", strip=True)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, blocking_fetch)

    async def _build_vector_index(self, text: str):
        def blocking_build():
            doc = Document(text=text)
            node_parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=256)
            nodes = node_parser.get_nodes_from_documents([doc])
            return VectorStoreIndex(nodes, embed_model=self.embed_model, show_progress=False)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, blocking_build)

    async def query_vector_index(self, query: str, top_k: int = 3) -> str:
        def blocking_query():
            query_engine = self.vector_index.as_query_engine(
                llm=self.llm, similarity_top_k=top_k, response_mode="compact"
            )
            return str(query_engine.query(query))

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, blocking_query)

    async def setup_vector_index_from_url(self, url: str) -> None:
        logger.info("Fetching and indexing filing from URL…")
        self.source_text = await self._fetch_and_parse_10q(url)
        self.vector_index = await self._build_vector_index(self.source_text)
        logger.info("Vector index ready from URL.")

    async def setup_vector_index_from_text(self, text: str) -> None:
        logger.info("Indexing uploaded document text…")
        self.source_text = text
        self.vector_index = await self._build_vector_index(self.source_text)
        logger.info("Vector index ready from uploaded text.")

    async def run_single_agent(self, cfg: Dict[str, Any], reflection_agent: ChatAgent, credential, project_endpoint: str):
        try:
            client = await create_client(credential, project_endpoint, cfg["id"])
            try:
                agent = ChatAgent(chat_client=client, instructions=cfg["base_prompt"])

                rag_context = await self.query_vector_index(cfg["rag_prompt"], top_k=5)
                prompt_input = cfg["base_prompt"].format(rag_context=rag_context)

                _, _, final_output = await run_generator_with_reflection(
                    agent, reflection_agent, prompt_input, cfg["revise_prompt"]
                )
                logger.info("Finished %s", cfg["name"])
                return cfg["name"], final_output.strip()
            finally:
                # Ensure underlying aiohttp sessions are closed
                if hasattr(client, "aclose"):
                    await client.aclose()
                elif hasattr(client, "close") and asyncio.iscoroutinefunction(client.close):
                    await client.close()
        except Exception as e:
            logger.exception("%s failed", cfg["name"])
            return cfg["name"], f"Error: {e}"

    async def run_agent_pipeline(self) -> Dict[str, Any]:
        async with AzureCliCredential() as credential:
            project_endpoint = os.getenv(
                "AZURE_AI_PROJECT_ENDPOINT",
                "https://aif-creditai-qa-001.services.ai.azure.com/api/projects/aif-proj-creditai-qa-001",
            )
            reflection_client = await create_client(
                credential, project_endpoint, "asst_c2s7pi3lsz5ZtbOfukp74bId"
            )
            try:
                reflection_agent = ChatAgent(chat_client=reflection_client, instructions=REFLECTION_PROMPT)

                tasks = [
                    self.run_single_agent(cfg, reflection_agent, credential, project_endpoint)
                    for cfg in GENERATOR_CONFIGS
                ]
                results_list = await asyncio.gather(*tasks)
                results = {k: v for k, v in results_list}
                return results
            finally:
                # Close reflection client to avoid unclosed session warnings
                if hasattr(reflection_client, "aclose"):
                    await reflection_client.aclose()
                elif hasattr(reflection_client, "close") and asyncio.iscoroutinefunction(reflection_client.close):
                    await reflection_client.close()

    # Helper functions (reorganized only)
async def create_client(credential, project_endpoint: str, agent_id: str):
    return AzureAIAgentClient(
        async_credential=credential,
        project_endpoint=project_endpoint,
        agent_id=agent_id,
    )

async def run_generator_with_reflection(
    generator_agent: ChatAgent,
    reflection_agent: ChatAgent,
    input_text: str,
    revision_prompt: str,
):
    thread = generator_agent.get_new_thread()

    # Step 1: Generate initial output
    gen_response = await generator_agent.run(input_text, thread=thread)

    # Step 2: Reflect on the generated output
    reflect_input = (
        f"{REFLECTION_PROMPT}\n\n"
        f"Generator Output:\n{getattr(gen_response, 'text', '')}"
    )
    reflect_response = await reflection_agent.run(reflect_input, thread=thread)

    # Step 3: Revise based on reflection
    revise_input = (
        f"{revision_prompt}\n"
        f"Reflection feedback:\n{getattr(reflect_response, 'text', '')}"
    )
    revised_response = await generator_agent.run(revise_input, thread=thread)

    return (
        getattr(gen_response, "text", "").strip(),
        getattr(reflect_response, "text", "").strip(),
        getattr(revised_response, "text", "").strip(),
    )

# File parsing (reorganized from risk-11)
def extract_text_from_file(file_obj, filename: Optional[str] = None, file_type: Optional[str] = None) -> str:
    try:
        ftype = file_type or getattr(file_obj, "type", None) or _infer_mime_from_name(filename)
        buffer = None
        if isinstance(file_obj, io.BytesIO):
            buffer = file_obj
            try:
                buffer.seek(0)
            except Exception:
                pass
        elif hasattr(file_obj, "read"):
            buffer = io.BytesIO(file_obj.read())
            buffer.seek(0)
        elif isinstance(file_obj, (bytes, bytearray)):
            buffer = io.BytesIO(file_obj)
            buffer.seek(0)
        else:
            return ""

        text = ""
        if ftype == "application/pdf":
            try:
                pdf_reader = PyPDF2.PdfReader(buffer)
            except Exception as pe:
                logger.error("Unable to read PDF: %s", pe)
                return ""
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        elif ftype == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = docx2txt.process(buffer)
        elif ftype == "text/plain":
            text = buffer.read().decode("utf-8", errors="ignore")
        else:
            logger.error("Unsupported file type: %s", ftype or "unknown")
            return ""
        return text.strip()
    except Exception as e:
        logger.exception("Error reading file")
        return ""

_def_ext_map = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
}

def _infer_mime_from_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    for ext, mime in _def_ext_map.items():
        if name.lower().endswith(ext):
            return mime
    return None

# SEC helper (reorganized only)
def build_latest_10q_url_from_mapping(ticker: str, mapping_json_path: str) -> Optional[str]:
    try:
        with open(mapping_json_path, "r") as f:
            mapping = json.load(f)
        df = pd.DataFrame(mapping["data"], columns=mapping["fields"])
        row = df[df["ticker"] == ticker]
        if row.empty:
            return None
        cik = str(int(row.iloc[0]["cik"]))

        headers = {"User-Agent": os.getenv("SEC_USER_AGENT", "creditai@example.com")}
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
        subs = requests.get(submissions_url, headers=headers, timeout=30).json()
        recent = pd.DataFrame(subs["filings"]["recent"])
        q = recent[recent.form == "10-Q"]
        if q.empty:
            return None
        acc_num = q.accessionNumber.values[0].replace("-", "")
        doc_name = q.primaryDocument.values[0]
        html_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_num}/{doc_name}"
        return html_url
    except Exception:
        logger.exception("Failed building 10-Q URL from mapping")
        return None

async def run_full_pipeline_from_url(url: str, similarity_top_k: int = 5) -> Dict[str, Any]:
    pipeline = FinancialRAGMAgenticSystem(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://ea-oai-sandbox.openai.azure.com"),
        api_key=os.getenv("AZURE_OPENAI_KEY", ""),
        embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
        openai_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"),
    )
    await pipeline.setup_vector_index_from_url(url)
    # similarity_top_k is used inside query_vector_index; kept consistent via call sites
    results = await pipeline.run_agent_pipeline()
    return results

async def run_full_pipeline_from_text(text: str, similarity_top_k: int = 5) -> Dict[str, Any]:
    pipeline = FinancialRAGMAgenticSystem(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://ea-oai-sandbox.openai.azure.com"),
        api_key=os.getenv("AZURE_OPENAI_KEY", ""),
        embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
        openai_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"),
    )
    await pipeline.setup_vector_index_from_text(text)
    results = await pipeline.run_agent_pipeline()
    return results