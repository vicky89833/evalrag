from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    LLM_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    DATABASE_URL: str

    EVALRAG_ENV: str = "local"
    EVALRAG_DATA_DIR: str = "./data"

    ANSWER_MODEL: str = "gemini-2.0-flash"
    JUDGE_MODEL: str = "gemini-2.0-flash"
    ROUTER_MODEL: str = "gemini-2.0-flash"
    GOLDEN_MODEL: str = "gemini-2.0-flash"
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


def make_llm_client() -> object:
    """Single source of truth for the LLM client. Routes through Gemini's
    OpenAI-compatible endpoint by default; flip LLM_BASE_URL + GEMINI_API_KEY →
    OPENAI_API_KEY in .env to swap providers without touching call sites."""
    from openai import OpenAI

    s = get_settings()
    api_key = s.GEMINI_API_KEY or s.OPENAI_API_KEY
    return OpenAI(api_key=api_key, base_url=s.LLM_BASE_URL)
