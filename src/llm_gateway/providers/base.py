from dataclasses import dataclass
from typing import AsyncIterator, Protocol

from llm_gateway.api.schemas import ChatCompletionRequest


@dataclass(frozen=True)
class ProviderResult:
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str = "stop"


@dataclass(frozen=True)
class ProviderStreamChunk:
    content: str | None = None
    model: str | None = None
    finish_reason: str | None = None


class ProviderError(Exception):
    """A sanitized upstream failure safe to expose through the gateway API."""


class ChatProvider(Protocol):
    async def complete(self, request: ChatCompletionRequest) -> ProviderResult: ...

    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ProviderStreamChunk]: ...
