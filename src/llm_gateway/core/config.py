from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or ``.env``."""

    upstream_provider: Literal["mock", "gemini", "groq"] = "mock"
    upstream_model: str = "mock-model"
    gemini_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    upstream_timeout_seconds: float = Field(default=30.0, gt=0, le=300)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def require_provider_credentials(self) -> "Settings":
        if (
            self.upstream_provider == "gemini"
            and (self.gemini_api_key is None or not self.gemini_api_key.get_secret_value().strip())
        ):
            raise ValueError("GEMINI_API_KEY is required when UPSTREAM_PROVIDER=gemini")
        if (
            self.upstream_provider == "groq"
            and (self.groq_api_key is None or not self.groq_api_key.get_secret_value().strip())
        ):
            raise ValueError("GROQ_API_KEY is required when UPSTREAM_PROVIDER=groq")
        if self.upstream_provider != "mock" and (
            not self.upstream_model.strip() or self.upstream_model == "mock-model"
        ):
            raise ValueError("Set UPSTREAM_MODEL to a model supported by the selected provider")
        return self
