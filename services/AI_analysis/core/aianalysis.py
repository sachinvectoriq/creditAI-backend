import os
import io
import re
import json
import asyncio
import logging
from typing import Optional, Dict, Any
import time
import uuid

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

# --- NEW: Application Insights telemetry ---
from applicationinsights import TelemetryClient

# ------------ USER PROVIDED INSTRUMENTATION ------------
INSTRUMENTATION_KEY = "d3e070dd-4b5c-46a3-820d-d77d328f280b"
APPINSIGHTS_CONNECTION_STRING = (
    "InstrumentationKey=d3e070dd-4b5c-46a3-820d-d77d328f280b;"
    "IngestionEndpoint=https://eastus2-3.in.applicationinsights.azure.com/;"
    "LiveEndpoint=https://eastus2.livediagnostics.monitor.azure.com/;"
    "ApplicationId=96c8dd19-99b0-4ba5-9d95-1691c5262498"
)
# -------------------------------------------------------

# initialize telemetry client
tc = TelemetryClient(INSTRUMENTATION_KEY)

# set up python logging to also forward to App Insights (trace events)
logger = logging.getLogger("ai-analysis-core")
logger.setLevel(logging.INFO)
# console handler (existing)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Small helper that sends structured events/traces to App Insights.
def ai_trace(message: str, severity="INFO", properties: Optional[Dict[str, Any]] = None):
    """
    Send a trace to both the logger and Application Insights.
    properties: optional dict to be stored as customDimensions.
    """
    if severity == "ERROR":
        logger.error(message)
    elif severity == "WARNING":
        logger.warning(message)
    else:
        logger.info(message)
    try:
        tc.track_trace(message, properties=properties or {})
        # also track an event for important milestones if provided
    except Exception:
        # don't let telemetry failures break the main app
        logger.debug("Failed to send telemetry trace to App Insights", exc_info=True)

def ai_event(name: str, properties: Optional[Dict[str, Any]] = None, metrics: Optional[Dict[str, float]] = None):
    """
    Track a custom event in Application Insights.
    """
    try:
        tc.track_event(name, properties=properties or {}, measurements=metrics or {})
    except Exception:
        logger.debug("Failed to send telemetry event", exc_info=True)

def ai_exception(exc: Exception, properties: Optional[Dict[str, Any]] = None):
    """
    Track exception details in App Insights.
    """
    try:
        # record exception details; applicationinsights will capture stack trace if provided
        tc.track_exception()
        # also log one structured trace
        tc.track_trace(str(exc), properties=properties or {})
    except Exception:
        logger.debug("Failed while sending exception telemetry", exc_info=True)


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

Extract the most recent quarter’s figures into an internal JSON with fields: Company, Report_Date; Liquidity (Cash_and_Equivalents, Total_Current_Assets, Total_CURRENT_Liabilities, Current_Ratio = Assets/Liabilities, Operating_Cash_Flow, Liquidity_Runway_Months = if OCF<0 then Cash / |OCF/12| else “Not applicable”); Leverage (Total_Debt, Shareholders_Equity, Debt_to_Equity = Debt/Equity, Debt_Maturities {2025…2029_and_beyond}, Undrawn_Facilities); Profitability (Revenue, Operating_Income, Net_Income, Operating_Margin = OpInc/Revenue); Cash_Flow (Operating_Cash_Flow, Capex, Free_Cash_Flow = OCF–Capex); Commitments_Contingencies (Purchase_Obligations, Legal_Tax_Exposure).

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
ogger = logging.getLogger(__name__)

# ======================================================
# 🔹 RAG + Agentic System (API Key + AAD separation)
# ======================================================
class FinancialRAGMAgenticSystem:
    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        embedding_deployment: str,
        openai_deployment: str,
        api_version: str = "2024-02-01",
    ):
        # create a correlation id for this instance (can be overridden per pipeline run)
        self.instance_correlation_id = str(uuid.uuid4())

        # telemetry: note creation
        ai_trace(
            "Initializing FinancialRAGMAgenticSystem",
            properties={
                "correlation_id": self.instance_correlation_id,
                "azure_endpoint": azure_endpoint,
                "embedding_deployment": embedding_deployment,
                "openai_deployment": openai_deployment,
            },
        )

        # ✅ API-key based auth for RAG models (same as code 2)
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

        # Register globally in LlamaIndex
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm

        # Thread executor + placeholders
        self.vector_index = None
        self.source_text = ""
        self.executor = ThreadPoolExecutor(max_workers=10)

    # ======================================================
    # 🔸 RAG setup and retrieval
    # ======================================================
    async def _fetch_and_parse_10q(self, url: str) -> str:
        start_ts = time.time()
        run_cid = str(uuid.uuid4())
        ai_trace("fetch_and_parse_10q_start", properties={"correlation_id": run_cid, "url": url})
        def blocking_fetch() -> str:
            headers = {
                "User-Agent": "Sachin Bhusnurmath (sachin@example.com)",
                "Accept-Encoding": "gzip, deflate",
                "Accept-Language": "en-US,en;q=0.9",
            }
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            return soup.get_text(separator="\n", strip=True)

        try:
            text = await asyncio.get_event_loop().run_in_executor(self.executor, blocking_fetch)
            elapsed = time.time() - start_ts
            ai_trace("fetch_and_parse_10q_success", properties={"correlation_id": run_cid, "elapsed_s": elapsed, "url": url})
            return text
        except Exception as e:
            elapsed = time.time() - start_ts
            ai_exception(e, properties={"correlation_id": run_cid, "elapsed_s": elapsed, "url": url})
            raise

    async def _build_vector_index(self, text: str):
        start_ts = time.time()
        run_cid = str(uuid.uuid4())
        ai_trace("build_vector_index_start", properties={"correlation_id": run_cid, "text_len": len(text)})
        def blocking_build():
            doc = Document(text=text)
            node_parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=256)
            nodes = node_parser.get_nodes_from_documents([doc])
            return VectorStoreIndex(nodes, embed_model=self.embed_model, show_progress=False)

        try:
            index = await asyncio.get_event_loop().run_in_executor(self.executor, blocking_build)
            self.vector_index = index
            elapsed = time.time() - start_ts
            ai_trace("build_vector_index_success", properties={"correlation_id": run_cid, "elapsed_s": elapsed, "node_count": len(index._store._nodes) if hasattr(index, "_store") else None})
            return index
        except Exception as e:
            elapsed = time.time() - start_ts
            ai_exception(e, properties={"correlation_id": run_cid, "elapsed_s": elapsed})
            raise

    async def query_vector_index(self, query: str, top_k: int = 3) -> str:
        start_ts = time.time()
        run_cid = str(uuid.uuid4())
        ai_trace("query_vector_index_start", properties={"correlation_id": run_cid, "query": query, "top_k": top_k})
        def blocking_query():
            query_engine = self.vector_index.as_query_engine(
                llm=self.llm, similarity_top_k=top_k, response_mode="compact"
            )
            return str(query_engine.query(query))

        try:
            result = await asyncio.get_event_loop().run_in_executor(self.executor, blocking_query)
            elapsed = time.time() - start_ts
            ai_trace("query_vector_index_success", properties={"correlation_id": run_cid, "elapsed_s": elapsed})
            # also log RAG context length to App Insights
            ai_event("rag_query", properties={"correlation_id": run_cid, "query_len": len(query), "top_k": top_k})
            return result
        except Exception as e:
            elapsed = time.time() - start_ts
            ai_exception(e, properties={"correlation_id": run_cid, "elapsed_s": elapsed})
            raise

    async def setup_vector_index_from_url(self, url: str) -> None:
        ai_trace("setup_vector_index_from_url_start", properties={"url": url})
        print("🌐 Fetching and indexing filing...")
        self.source_text = await self._fetch_and_parse_10q(url)
        await self._build_vector_index(self.source_text)
        print("✅ Vector index ready.\n")
        ai_trace("setup_vector_index_from_url_done", properties={"url": url})

    async def setup_vector_index_from_text(self, text: str) -> None:
        ai_trace("setup_vector_index_from_text_start", properties={"text_len": len(text)})
        print("🧾 Indexing uploaded text...")
        self.source_text = text
        await self._build_vector_index(self.source_text)
        print("✅ Vector index ready from text.\n")
        ai_trace("setup_vector_index_from_text_done", properties={"text_len": len(text)})

    # ======================================================
    # 🔸 Agentic pipeline (AAD token–based)
    # ======================================================
    async def run_single_agent(self, cfg: Dict[str, Any], reflection_agent, credential, project_endpoint: str):
        run_cid = str(uuid.uuid4())
        start_time = time.time()
        ai_trace("run_single_agent_start", properties={"correlation_id": run_cid, "agent_name": cfg.get("name"), "agent_id": cfg.get("id")})
        try:
            client = await create_client(credential, project_endpoint, cfg["id"])
            agent = ChatAgent(chat_client=client, instructions=cfg["base_prompt"])

            rag_context = await self.query_vector_index(cfg["rag_prompt"], top_k=5)
            prompt_input = cfg["base_prompt"].format(rag_context=rag_context)

            # Log agent input to App Insights
            ai_event(
                "agent_input",
                properties={
                    "correlation_id": run_cid,
                    "agent_name": cfg.get("name"),
                    "agent_id": cfg.get("id"),
                    "prompt_input": (prompt_input[:4000] if prompt_input else "")
                },
            )

            _, _, final_output = await run_generator_with_reflection(
                agent, reflection_agent, prompt_input, cfg["revise_prompt"]
            )

            elapsed = time.time() - start_time
            ai_event(
                "agent_output",
                properties={
                    "correlation_id": run_cid,
                    "agent_name": cfg.get("name"),
                    "agent_id": cfg.get("id"),
                    "output_len": len(final_output) if final_output else 0
                },
            )

            # store the full output as a trace (safeguard length)
            ai_trace(
                f"agent_output_full:{cfg.get('name')}",
                properties={
                    "correlation_id": run_cid,
                    "agent_name": cfg.get("name"),
                    "output": (final_output[:8000] if final_output else "")
                },
            )

            print(f"✅ Finished {cfg['name']}")
            return cfg["name"], final_output.strip()
        except Exception as e:
            ai_exception(e, properties={"correlation_id": run_cid, "agent_name": cfg.get("name")})
            print(f"❌ {cfg['name']} failed: {e}")
            return cfg["name"], f"Error: {e}"

    async def run_agent_pipeline(self, company_url: Optional[str] = None):
        # create pipeline-level correlation id
        pipeline_cid = str(uuid.uuid4())
        ai_trace("run_agent_pipeline_start", properties={"correlation_id": pipeline_cid, "company_url": company_url})
        if company_url:
            await self.setup_vector_index_from_url(company_url)

        # ✅ EXACT same AAD initialization as code 2
        async with AzureCliCredential() as credential:
            project_endpoint = (
                "https://aif-creditai-qa-001.services.ai.azure.com/api/projects/aif-proj-creditai-qa-001"
            )

            reflection_client = await create_client(
                credential, project_endpoint, "asst_c2s7pi3lsz5ZtbOfukp74bId"
            )
            reflection_agent = ChatAgent(chat_client=reflection_client, instructions=REFLECTION_PROMPT)

            tasks = [
                self.run_single_agent(cfg, reflection_agent, credential, project_endpoint)
                for cfg in GENERATOR_CONFIGS
            ]

            results_list = await asyncio.gather(*tasks)
            results = {k: v for k, v in results_list}

            print("\n✅ FINAL RESULTS:\n")
            print(json.dumps(results, indent=2, ensure_ascii=False))

            ai_event("pipeline_completed", properties={"correlation_id": pipeline_cid, "result_count": len(results)})
            # flush telemetry to App Insights
            try:
                tc.flush()
            except Exception:
                logger.debug("Failed to flush telemetry", exc_info=True)

            return results


# ======================================================
# 🔹 Helper functions
# ======================================================
async def create_client(credential, project_endpoint: str, agent_id: str):
    # log client creation
    cid = str(uuid.uuid4())
    ai_trace("create_client", properties={"correlation_id": cid, "project_endpoint": project_endpoint, "agent_id": agent_id})
    return AzureAIAgentClient(
        async_credential=credential,
        project_endpoint=project_endpoint,
        agent_id=agent_id,
    )


async def run_generator_with_reflection(
    generator_agent,
    reflection_agent,
    input_text: str,
    revision_prompt: str,
):
    run_cid = str(uuid.uuid4())
    ai_trace("run_generator_with_reflection_start", properties={"correlation_id": run_cid})

    thread = generator_agent.get_new_thread()
    # Log generator run input
    ai_event("generator_input", properties={"correlation_id": run_cid, "input_len": len(input_text) if input_text else 0})

    gen_response = await generator_agent.run(input_text, thread=thread)
    ai_event("generator_response", properties={"correlation_id": run_cid, "response_len": len(gen_response.text) if getattr(gen_response, "text", None) else 0})
    # Limit what we include in logs to avoid excessive size
    ai_trace("generator_response_text", properties={"correlation_id": run_cid, "text_snippet": (gen_response.text[:4000] if getattr(gen_response, "text", None) else "")})

    reflect_input = f"{REFLECTION_PROMPT}\n\nGenerator Output:\n{gen_response.text}"
    ai_event("reflection_input", properties={"correlation_id": run_cid, "input_len": len(reflect_input)})
    reflect_response = await reflection_agent.run(reflect_input, thread=thread)
    ai_event("reflection_response", properties={"correlation_id": run_cid, "response_len": len(reflect_response.text) if getattr(reflect_response, "text", None) else 0})
    ai_trace("reflection_response_text", properties={"correlation_id": run_cid, "text_snippet": (reflect_response.text[:2000] if getattr(reflect_response, "text", None) else "")})

    revise_input = f"{revision_prompt}\nReflection feedback:\n{reflect_response.text}"
    ai_event("revision_input", properties={"correlation_id": run_cid, "input_len": len(revise_input)})
    revised_response = await generator_agent.run(revise_input, thread=thread)
    ai_event("revision_response", properties={"correlation_id": run_cid, "response_len": len(revised_response.text) if getattr(revised_response, "text", None) else 0})
    ai_trace("revision_response_text", properties={"correlation_id": run_cid, "text_snippet": (revised_response.text[:4000] if getattr(revised_response, "text", None) else "")})

    ai_trace("run_generator_with_reflection_done", properties={"correlation_id": run_cid})
    return gen_response.text, reflect_response.text, revised_response.text


# ======================================================
# 🔹 Entry points
# ======================================================
async def run_full_pipeline_from_url(url: str, similarity_top_k: int = 5) -> Dict[str, Any]:
    pipeline = FinancialRAGMAgenticSystem(
        azure_endpoint="https://ea-oai-sandbox.openai.azure.com",
        api_key="2f6e41aa534f49908feb01c6de771d6b",
        embedding_deployment="text-embedding-ada-002",
        openai_deployment="gpt-4o",
    )
    await pipeline.setup_vector_index_from_url(url)
    res = await pipeline.run_agent_pipeline(company_url=None)
    # flush telemetry after run
    try:
        tc.flush()
    except Exception:
        logger.debug("Failed to flush telemetry at pipeline end", exc_info=True)
    return res


async def run_full_pipeline_from_text(text: str, similarity_top_k: int = 5) -> Dict[str, Any]:
    pipeline = FinancialRAGMAgenticSystem(
        azure_endpoint="https://ea-oai-sandbox.openai.azure.com",
        api_key="2f6e41aa534f49908feb01c6de771d6b",
        embedding_deployment="text-embedding-ada-002",
        openai_deployment="gpt-4o",
    )
    await pipeline.setup_vector_index_from_text(text)
    res = await pipeline.run_agent_pipeline(company_url=None)
    try:
        tc.flush()
    except Exception:
        logger.debug("Failed to flush telemetry at pipeline end", exc_info=True)
    return res

# File parsing (reorganized from risk-11)
def extract_text_from_file(file_obj, filename: Optional[str] = None, file_type: Optional[str] = None) -> str:
    run_cid = str(uuid.uuid4())
    ai_trace("extract_text_from_file_start", properties={"correlation_id": run_cid, "filename": filename})
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
            ai_trace("extract_text_from_file_unsupported_type", properties={"correlation_id": run_cid, "file_type": ftype or "unknown"})
            return ""

        text = ""
        if ftype == "application/pdf":
            try:
                pdf_reader = PyPDF2.PdfReader(buffer)
            except Exception as pe:
                logger.error("Unable to read PDF: %s", pe)
                ai_exception(pe, properties={"correlation_id": run_cid, "filename": filename})
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
            ai_trace("extract_text_from_file_unsupported_type", properties={"correlation_id": run_cid, "file_type": ftype or "unknown"})
            return ""
        ai_trace("extract_text_from_file_success", properties={"correlation_id": run_cid, "text_len": len(text)})
        return text.strip()
    except Exception as e:
        logger.exception("Error reading file")
        ai_exception(e, properties={"correlation_id": run_cid})
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
    run_cid = str(uuid.uuid4())
    ai_trace("build_latest_10q_url_from_mapping_start", properties={"correlation_id": run_cid, "ticker": ticker})
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
        ai_trace("build_latest_10q_url_from_mapping_success", properties={"correlation_id": run_cid, "html_url": html_url})
        return html_url
    except Exception:
        logger.exception("Failed building 10-Q URL from mapping")
        ai_exception(Exception("Failed building 10-Q URL from mapping"), properties={"correlation_id": run_cid})
        return None
