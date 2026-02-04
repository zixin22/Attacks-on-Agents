"""
Quick API connectivity check for the local OpenAI-compatible endpoint.
Reads key from OpenAI_api_key.txt or OPENAI_API_KEY env var.
"""

import os
from pathlib import Path

import openai


def load_api_key() -> str:
    key_path = Path(__file__).parent.parent / "OpenAI_api_key.txt"
    if key_path.exists():
        return key_path.read_text(encoding="utf-8").strip()
    return os.getenv("OPENAI_API_KEY", "").strip()


def main():
    api_key = load_api_key()
    if not api_key:
        raise SystemExit("No API key found in OpenAI_api_key.txt or OPENAI_API_KEY.")

    base_url = "http://152.53.53.64:3000/v1"
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    # Minimal chat completion to verify connectivity/auth.
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=5,
        temperature=0,
    )
    print("OK:", resp.choices[0].message.content)


if __name__ == "__main__":
    main()















