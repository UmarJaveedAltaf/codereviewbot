from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM
    GOOGLE_API_KEY: str

    # GitHub
    GITHUB_TOKEN: str
    GITHUB_WEBHOOK_SECRET: str

    # Slack
    SLACK_WEBHOOK_URL: str = ""

    # Vector store
    CHROMA_PERSIST_DIR: str = "./.chroma"
    CHROMA_COLLECTION: str = "code_reviews"

    # App
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # LLM model selection
    GEMINI_MODEL: str = "gemini-1.5-flash"
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"


settings = Settings()
