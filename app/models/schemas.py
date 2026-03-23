from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="Immigration / visa question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")


class SourceReference(BaseModel):
    title: str
    url: str
    category: str
    score: float = Field(ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceReference]
    model_used: str


class IngestRequest(BaseModel):
    force_reingest: bool = Field(default=False, description="Re-ingest even if collection already exists")


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    sources_processed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    collection_exists: bool
    total_chunks: Optional[int] = None
    embedding_model: str
