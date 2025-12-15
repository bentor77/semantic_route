from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.1-70b-versatile"
    
    QDRANT_URL: str
    QDRANT_API_KEY: str
    QDRANT_COLLECTION: str = "semantic-router-index"
    
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
