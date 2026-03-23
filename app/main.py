"""
International Student AI Assistant – FastAPI Application
RAG-based Q&A for immigration and visa queries from official USCIS sources.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.qa_engine import answer_question
from app.ingestion.pipeline import (
    run_ingestion_pipeline,
    get_qdrant_client,
    collection_count,
)
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    IngestRequest,
    IngestResponse,
    HealthResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting International Student AI Assistant...")
    logger.info("Qdrant host: %s:%s", settings.qdrant_host, settings.qdrant_port)
    logger.info("Embedding model: %s", settings.embedding_model)
    yield
    logger.info("Shutting down...")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="International Student AI Assistant",
    description=(
        "RAG-based Q&A system for international students. "
        "Answers immigration and visa queries from official USCIS sources."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "International Student AI Assistant",
        "description": "Ask me anything about F-1 visas, OPT, STEM OPT, H-1B, and US immigration.",
        "endpoints": {
            "ask": "POST /ask",
            "ingest": "POST /ingest",
            "health": "GET /health",
            "docs": "GET /docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check system health: API, Qdrant connection, and collection status."""
    try:
        client = get_qdrant_client()
        client.get_collections()
        qdrant_ok = True
    except Exception as exc:
        logger.warning("Qdrant connection failed: %s", exc)
        qdrant_ok = False

    try:
        count = collection_count(client) if qdrant_ok else None
        col_exists = count is not None
    except Exception:
        count = None
        col_exists = False

    return HealthResponse(
        status="healthy" if qdrant_ok else "degraded",
        qdrant_connected=qdrant_ok,
        collection_exists=col_exists,
        total_chunks=count,
        embedding_model=settings.embedding_model,
    )


@app.post("/ask", response_model=QueryResponse, tags=["Q&A"])
async def ask_question(request: QueryRequest):
    """
    Ask an immigration or visa question.

    Uses semantic vector search (Qdrant) to retrieve relevant USCIS policy chunks,
    then generates a grounded answer via OpenAI GPT.
    """
    try:
        result = answer_question(
            question=request.question,
            top_k=request.top_k,
        )
        return result
    except Exception as exc:
        logger.error("Error answering question: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(exc)}")


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_documents(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger the data ingestion pipeline.

    Scrapes USCIS official pages → cleans → chunks → embeds (sentence-transformers)
    → indexes into Qdrant vector database.

    Set `force_reingest=true` to wipe and re-index existing data.
    """
    try:
        result = run_ingestion_pipeline(force=request.force_reingest)
        return IngestResponse(**result)
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(exc)}")


@app.get("/sources", tags=["Info"])
async def list_sources():
    """Return the list of official USCIS sources used in the knowledge base."""
    from app.ingestion.sources import USCIS_SOURCES
    return {
        "total": len(USCIS_SOURCES),
        "sources": USCIS_SOURCES,
    }


@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "docs": "/docs"},
    )
