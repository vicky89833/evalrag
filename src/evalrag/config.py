from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ANTHROPIC_API_KEY: str
    OPENAI_API_KEY: str
    DATABASE_URL: str

    EVALRAG_ENV: str = "local"
    EVALRAG_DATA_DIR: str = "./data"

    ANSWER_MODEL: str = "claude-sonnet-4-6"
    JUDGE_MODEL: str = "gpt-4o-mini"
    ROUTER_MODEL: str = "gpt-4o-mini"
    GOLDEN_MODEL: str = "gpt-4o-mini"
    EMBED_MODEL: str = "BAAI/bge-large-en-v1.5"
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"

    EMBED_DIM: int = 1024
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    TOP_K_RETRIEVE: int = 20
    TOP_K_RERANK: int = 5

    MAX_UPLOAD_MB: int = 50
    MAX_CTX_TOKENS: int = 8000
    PER_DOC_COST_CAP_USD: float = 1.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
