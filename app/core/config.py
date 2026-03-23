from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = "your-key-here"
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "uscis_knowledge"

    # Sentence Transformers (local fallback)
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: str = "development"

    # Ingestion
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_results: int = 5

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
