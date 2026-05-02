from evalrag.config import Settings


def test_settings_pins_models_and_dims():
    s = Settings(_env_file=None, ANTHROPIC_API_KEY="x", OPENAI_API_KEY="y",
                 DATABASE_URL="postgresql+psycopg://u:p@h/db")
    assert s.ANSWER_MODEL == "claude-sonnet-4-6"
    assert s.JUDGE_MODEL == "gpt-4o-mini"
    assert s.EMBED_MODEL == "BAAI/bge-large-en-v1.5"
    assert s.EMBED_DIM == 1024
    assert s.CHUNK_SIZE == 512
    assert s.CHUNK_OVERLAP == 64
    assert s.TOP_K_RETRIEVE == 20
    assert s.TOP_K_RERANK == 5
