from collections.abc import AsyncIterator

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI

from llm_gateway.api.schemas import ChatCompletionRequest
from llm_gateway.providers.base import ProviderError, ProviderResult, ProviderStreamChunk


PROVIDER_BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1/",
}


class OpenAICompatibleProvider:
    """Adapter for providers that accept OpenAI chat-completion requests."""

    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        try:
            base_url = PROVIDER_BASE_URLS[provider]
        except KeyError as exc:
            raise ValueError(f"Unsupported OpenAI-compatible provider: {provider}") from exc

        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=0,
        )

    def _payload(self, request: ChatCompletionRequest) -> dict[str, object]:
        # Only forward fields implemented by this gateway's public schema.
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [message.model_dump() for message in request.messages],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        return payload

    async def complete(self, request: ChatCompletionRequest) -> ProviderResult:
        payload = self._payload(request)

        try:
            response = await self.client.chat.completions.create(**payload)  # type: ignore[arg-type]
        except (APIConnectionError, APITimeoutError, APIError) as exc:
            # Do not return provider response bodies or credentials to the caller.
            raise ProviderError("The upstream provider request failed") from exc

        if not response.choices:
            raise ProviderError("The upstream provider returned no choices")

        choice = response.choices[0]
        content = choice.message.content
        if content is None:
            raise ProviderError("The upstream provider returned no text content")

        usage = response.usage
        return ProviderResult(
            content=content,
            model=response.model or self.model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            finish_reason=choice.finish_reason or "stop",
        )

    async def _stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ProviderStreamChunk]:
        payload = self._payload(request)
        upstream_stream = None
        try:
            upstream_stream = await self.client.chat.completions.create(
                **payload, stream=True  # type: ignore[arg-type]
            )
            async for chunk in upstream_stream:
                model = chunk.model or self.model
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                content = choice.delta.content
                # Some compatible providers send non-text delta parts; this gateway
                # currently supports text-only chat messages.
                if isinstance(content, str) and content:
                    yield ProviderStreamChunk(content=content, model=model)
                if choice.finish_reason:
                    yield ProviderStreamChunk(
                        model=model,
                        finish_reason=choice.finish_reason,
                    )
        except (APIConnectionError, APITimeoutError, APIError) as exc:
            raise ProviderError("The upstream provider request failed") from exc
        finally:
            if upstream_stream is not None:
                await upstream_stream.close()

    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ProviderStreamChunk]:
        return self._stream(request)

    async def close(self) -> None:
        await self.client.close()
