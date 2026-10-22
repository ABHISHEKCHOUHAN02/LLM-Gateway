from llm_gateway.api.schemas import ChatCompletionRequest
from collections.abc import AsyncIterator

from llm_gateway.providers.base import ProviderResult, ProviderStreamChunk


class MockProvider:
    """Deterministic provider used to exercise the gateway without credentials."""

    async def complete(self, request: ChatCompletionRequest) -> ProviderResult:
        last_user_message = next(
            (message.content for message in reversed(request.messages) if message.role == "user"),
            "",
        )
        return ProviderResult(
            content=f"Mock response: {last_user_message}",
            model=request.model,
        )

    async def _stream(self, request: ChatCompletionRequest) -> AsyncIterator[ProviderStreamChunk]:
        result = await self.complete(request)
        # Split into short chunks to make the mock useful for client-side stream demos.
        text = result.content
        for offset in range(0, len(text), 12):
            yield ProviderStreamChunk(content=text[offset : offset + 12], model=result.model)
        yield ProviderStreamChunk(model=result.model, finish_reason="stop")

    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ProviderStreamChunk]:
        return self._stream(request)
