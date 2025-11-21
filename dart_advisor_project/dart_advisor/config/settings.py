"""Application settings"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")  # Optional for lite mode
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: str = Field("logs/dart_advisor.log", env="LOG_FILE")
    output_dir: Path = Field(Path("output"), env="OUTPUT_DIR")
    cache_dir: Path = Field(Path(".cache"), env="CACHE_DIR")
    claude_model: str = Field("claude-sonnet-4-20250514", env="CLAUDE_MODEL")
    max_tokens: int = Field(8192, env="MAX_TOKENS")
    temperature: float = Field(0.3, env="TEMPERATURE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

    def has_api_key(self) -> bool:
        """Check if API key is configured"""
        return bool(self.anthropic_api_key and self.anthropic_api_key != "your_api_key_here")

_settings = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
