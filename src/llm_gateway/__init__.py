"""OpenAI-compatible LLM gateway."""


def main() -> None:
    """Run the development server (``llm-gateway`` command)."""
    import uvicorn

    uvicorn.run("llm_gateway.main:app", host="127.0.0.1", port=8000, reload=False)
