# LLM Gateway

An OpenAI-compatible FastAPI gateway. Phase 1 provides a deterministic mock
upstream; Phase 2 adds real Gemini and Groq upstreams through their
OpenAI-compatible chat-completions APIs. The gateway returns OpenAI-shaped JSON
and SSE responses, while the selected provider generates the answer.

## Run with the mock provider

Install the project and start the service:

```powershell
python -m pip install -e .
llm-gateway
```

The server listens on `http://127.0.0.1:8000`. For local development, you can
also run `uvicorn llm_gateway.main:app --reload --app-dir src`.

## Run with Gemini

Create a `.env` file in the project root (do not commit it):

```dotenv
UPSTREAM_PROVIDER=gemini
UPSTREAM_MODEL=your-supported-gemini-model
GEMINI_API_KEY=your-gemini-api-key
```

For Groq, set `UPSTREAM_PROVIDER=groq`, `UPSTREAM_MODEL` to a model available
to your account, and `GROQ_API_KEY` instead. The service uses the selected
provider's credential only. `.env` is ignored by Git.

## Endpoints

- `GET /health`
- `GET /v1/models` — lists the configured upstream model
- `POST /v1/chat/completions` — JSON or streaming chat completion (`stream: true`)

The gateway currently exposes one configured model; requests naming any other
model receive a model-not-found error. Non-streaming provider failures are
returned as a sanitized `502` error. Streaming provider failures are sent as an
OpenAI-shaped SSE error event because the HTTP response has already started.

## Try a request

```powershell
curl.exe http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"your-supported-gemini-model","messages":[{"role":"user","content":"Hello"}]}'
```

The OpenAI Python SDK can be used as a client by setting `base_url` to
`http://127.0.0.1:8000/v1`. In this setup the SDK sends the request to this
gateway; Gemini or Groq remains the upstream model provider.

The runnable SDK example supports both JSON and streaming responses:

```powershell
python examples/openai_sdk_client.py
python examples/openai_sdk_client.py --stream
```

Set `GATEWAY_MODEL` to the configured model when using Gemini or Groq. The
gateway API key is a placeholder for this phase because gateway authentication
is added later.
