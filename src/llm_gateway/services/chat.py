import time
import uuid
import json
from collections.abc import AsyncIterator

from llm_gateway.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Usage,
)
from llm_gateway.providers.base import ChatProvider, ProviderError


def _estimate_tokens(text: str) -> int:
    """Rough placeholder for Phase 1; provider token counts will replace this."""
    return max(1, len(text.split())) if text else 0


class ChatService:
    def __init__(self, provider: ChatProvider) -> None:
        self.provider = provider

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        result = await self.provider.complete(request)
        prompt = " ".join(message.content for message in request.messages)
        prompt_tokens = _estimate_tokens(prompt)
        completion_tokens = _estimate_tokens(result.content)
        prompt_tokens = result.prompt_tokens if result.prompt_tokens is not None else prompt_tokens
        completion_tokens = (
            result.completion_tokens
            if result.completion_tokens is not None
            else completion_tokens
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            created=int(time.time()),
            model=result.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=result.content),
                    finish_reason=result.finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[str]:
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        def event(delta: dict[str, str], finish_reason: str | None = None) -> str:
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish_reason}
                ],
            }
            return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        yield event({"role": "assistant"})
        finish_reason = "stop"
        try:
            async for provider_chunk in self.provider.stream(request):
                if provider_chunk.content:
                    yield event({"content": provider_chunk.content})
                if provider_chunk.finish_reason:
                    finish_reason = provider_chunk.finish_reason
        except ProviderError as exc:
            error = {
                "error": {
                    "message": str(exc),
                    "type": "upstream_error",
                    "param": None,
                    "code": "provider_error",
                }
            }
            yield f"data: {json.dumps(error)}\n\n"
            yield "data: [DONE]\n\n"
            return

        yield event({}, finish_reason)
        yield "data: [DONE]\n\n"
