"""Call the gateway with the official OpenAI Python SDK.

Run the gateway first, then:
    python examples/openai_sdk_client.py
    python examples/openai_sdk_client.py --stream
"""

import argparse
import os

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description="Example OpenAI SDK client for the gateway")
    parser.add_argument("--stream", action="store_true", help="request an SSE response")
    parser.add_argument("--prompt", default="Explain what an LLM gateway does in one sentence.")
    args = parser.parse_args()

    client = OpenAI(
        api_key=os.getenv("GATEWAY_API_KEY", "local-demo-key"),
        base_url=os.getenv("GATEWAY_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    response = client.chat.completions.create(
        model=os.getenv("GATEWAY_MODEL", "mock-model"),
        messages=[{"role": "user", "content": args.prompt}],
        stream=args.stream,
    )

    if args.stream:
        for chunk in response:
            text = chunk.choices[0].delta.content
            if text:
                print(text, end="", flush=True)
        print()
    else:
        print(response.choices[0].message.content)

    client.close()


if __name__ == "__main__":
    main()
