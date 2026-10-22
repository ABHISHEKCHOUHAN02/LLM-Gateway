from fastapi import Request
from fastapi.exceptions import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# this file makes failures look like OpenAI’s too. The SDK expects errors in the form {"error": {"message", "type", "code"}}. This file defines your own exception types and handlers that convert them (including FastAPI’s validation errors) into that shape with the correct status code. Without it, the SDK would see FastAPI’s default error format and behave oddly.

async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": "Invalid request body",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
                "details": [
                    {key: error[key] for key in ("loc", "msg", "type") if key in error}
                    for error in exc.errors()
                ],
            }
        },
    )


async def http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": detail.get("message", "Request failed"),
                "type": detail.get("type", "invalid_request_error"),
                "param": detail.get("param"),
                "code": detail.get("code"),
            }
        },
    )


async def provider_error_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={
            "error": {
                "message": str(exc),
                "type": "upstream_error",
                "param": None,
                "code": "provider_error",
            }
        },
    )
