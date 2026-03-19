"""
src/api/main.py
────────────────
FastAPI application entry point.
Provides REST endpoints for incident diagnosis, knowledge base
management, and system health monitoring.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import structlog
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.requests import Request
from starlette.responses import Response

from src.api.models import (
    DiagnoseRequest, DiagnoseResponse,
    IndexRequest, IndexResponse,
    HealthResponse, StatsResponse,
)
from src.api.routes import create_router
from src.diagnosis.diagnoser import IncidentDiagnoser, IncidentContext
from src.diagnosis.llm_client import LLMClient
from src.indexing.chunker import DocumentChunker
from src.indexing.embedder import Embedder
from src.indexing.vector_store import FAISSVectorStore
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import SemanticRetriever
from src.retrieval.query_expander import HyDEQueryExpander

logger = structlog.get_logger(__name__)

# ── Prometheus metrics ──────────────────────────────────────────
REQUEST_COUNT = Counter(
    "aiops_requests_total", "Total diagnosis requests", ["status"]
)
REQUEST_LATENCY = Histogram(
    "aiops_request_latency_seconds", "Diagnosis request latency",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0]
)
RETRIEVAL_DOCS = Histogram(
    "aiops_retrieval_docs", "Number of documents retrieved per query",
    buckets=[0, 1, 2, 3, 5, 8, 10]
)


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── App State ───────────────────────────────────────────────────

class AppState:
    config: dict
    diagnoser: IncidentDiagnoser
    chunker: DocumentChunker
    embedder: Embedder
    vector_store: FAISSVectorStore


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise all components on startup."""
    logger.info("Initialising RAG-AIOps Framework")
    try:
        state.config = load_config()
        cfg = state.config

        # Build component chain
        state.vector_store = FAISSVectorStore({
            **cfg["vector_store"],
            "dimensions": cfg["embedding"]["dimensions"],
        })
        state.embedder = Embedder({
            **cfg["embedding"],
            **cfg["llm"],
            "cache": cfg.get("cache", {}),
        })
        reranker = CrossEncoderReranker(cfg["retrieval"]["reranker_model"])
        llm_client = LLMClient(cfg["llm"])
        query_expander = HyDEQueryExpander(llm_client) if cfg["retrieval"].get("hyde_enabled") else None

        retriever = SemanticRetriever(
            vector_store=state.vector_store,
            embedder=state.embedder,
            reranker=reranker,
            query_expander=query_expander,
            config=cfg["retrieval"],
        )

        state.diagnoser = IncidentDiagnoser(retriever, llm_client, cfg)
        state.chunker = DocumentChunker(cfg.get("chunking", {}))

        logger.info(
            "Framework initialised",
            index_size=state.vector_store.total_vectors,
        )
    except Exception as e:
        logger.error("Startup failed", error=str(e), exc_info=True)
        raise

    yield  # App running

    logger.info("Shutting down")


# ── FastAPI App ─────────────────────────────────────────────────

app = FastAPI(
    title="RAG-AIOps Incident Diagnosis API",
    description=(
        "REST API for the RAG-based AIOps framework described in:\n"
        "*Retrieval-Augmented Generation for Automated Incident Diagnosis "
        "in Cloud-Native DevOps Environments*\n"
        "Garg & Rengan, Journal of Software: Evolution and Process, Springer"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ───────────────────────────────────────────────────

@app.post("/v1/diagnose", response_model=DiagnoseResponse, tags=["Diagnosis"])
async def diagnose_incident(request: DiagnoseRequest):
    """
    **Primary endpoint** — Run the full RAG diagnosis pipeline on an incident.

    Returns structured root cause analysis, remediation steps, and
    confidence score. Corresponds to the 8-step workflow in the paper.
    """
    start = time.perf_counter()
    try:
        incident = IncidentContext(
            incident_id=request.incident_id,
            title=request.title,
            description=request.description,
            service=request.service,
            priority=request.priority,
            environment=request.environment,
            error_message=request.error_message,
            affected_components=request.affected_components or [],
            recent_deployments=request.recent_deployments or [],
        )
        result = state.diagnoser.diagnose(incident)

        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(time.perf_counter() - start)
        RETRIEVAL_DOCS.observe(len(result.retrieved_chunks))

        return DiagnoseResponse(**result.to_dict())

    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        logger.error("Diagnosis failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/health", response_model=HealthResponse, tags=["Operations"])
async def health_check():
    """System health check including vector store status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        vector_store_size=state.vector_store.total_vectors,
        index_loaded=state.vector_store.total_vectors > 0,
    )


@app.get("/v1/stats", response_model=StatsResponse, tags=["Operations"])
async def get_stats():
    """Return knowledge base and index statistics."""
    vs_stats = state.vector_store.stats()
    return StatsResponse(**vs_stats)


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
