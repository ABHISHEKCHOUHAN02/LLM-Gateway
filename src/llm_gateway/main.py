from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError

from llm_gateway.api.errors import (
    http_error_handler,
    provider_error_handler,
    validation_error_handler,
)
from llm_gateway.api.routes import router
from llm_gateway.core.config import Settings
from llm_gateway.providers.base import ProviderError
from llm_gateway.providers.mock import MockProvider
from llm_gateway.providers.openai_compatible import OpenAICompatibleProvider
from llm_gateway.services.chat import ChatService


def create_provider(settings: Settings) -> MockProvider | OpenAICompatibleProvider:
    if settings.upstream_provider == "mock":
        return MockProvider()

    api_key = (
        settings.gemini_api_key
        if settings.upstream_provider == "gemini"
        else settings.groq_api_key
    )
    assert api_key is not None  # Validated by Settings.
    return OpenAICompatibleProvider(
        provider=settings.upstream_provider,
        api_key=api_key.get_secret_value(),
        model=settings.upstream_model,
        timeout_seconds=settings.upstream_timeout_seconds,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    provider = create_provider(settings)
    app.state.gateway_model = settings.upstream_model
    app.state.chat_service = ChatService(provider)
    try:
        yield
    finally:
        if isinstance(provider, OpenAICompatibleProvider):
            await provider.close()


app = FastAPI(title="LLM Gateway", version="0.3.0", lifespan=lifespan)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(HTTPException, http_error_handler)
app.add_exception_handler(ProviderError, provider_error_handler)
app.include_router(router)
