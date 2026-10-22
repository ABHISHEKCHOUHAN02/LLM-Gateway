from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from llm_gateway.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelCard,
    ModelList,
)

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/v1/models", response_model=ModelList)
async def list_models(request: Request) -> ModelList:
    return ModelList(data=[ModelCard(id=request.app.state.gateway_model)])


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    chat_request: ChatCompletionRequest,
    request: Request,
) -> ChatCompletionResponse | StreamingResponse:
    if chat_request.model != request.app.state.gateway_model:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Model '{chat_request.model}' is not available",
                "type": "invalid_request_error",
                "code": "model_not_found",
            },
        )
    if chat_request.stream:
        return StreamingResponse(
            request.app.state.chat_service.stream(chat_request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return await request.app.state.chat_service.complete(chat_request)
