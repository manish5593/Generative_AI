#!/usr/bin/env python3
"""Local verification for llm_client.generate_response.

Usage:
    python test_llm_client.py

Reads OPENAI_API_KEY (and optional OPENAI_BASE_URL) from .env in this directory.
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit("OPENAI_API_KEY not found. Create a .env file with:\n  OPENAI_API_KEY=\"...\"")

from llm_client import generate_response

SAMPLE_CONTEXT = """[Source 1] Mission: Apollo 11 | Source: NASA_NTRS_19710015566 | Category: Mission Report
The lunar module Eagle landed in the Sea of Tranquility at 20:17:40 UTC on July 20, 1969.
Total time on the lunar surface was 21 hours, 36 minutes.

[Source 2] Mission: Apollo 11 | Source: NASA_NTRS_19710015566 | Category: Crew
Commander Neil A. Armstrong, Command Module Pilot Michael Collins, Lunar Module Pilot Edwin E. Aldrin Jr."""


def check(label: str, expectation: str, **kwargs) -> None:
    print(f"\n{'=' * 70}\n{label}\nExpect: {expectation}\n{'-' * 70}")
    print(generate_response(openai_key=api_key, **kwargs))


def main() -> None:
    print(f"key: {api_key[:7]}... | base_url: {os.getenv('OPENAI_BASE_URL', 'default (api.openai.com)')}")

    check(
        "TEST 1 - no retrieved context",
        "says it has no documents, then labels general knowledge",
        user_message="What was Apollo 11?",
        context="",
        conversation_history=[],
    )

    check(
        "TEST 2 - answerable from context",
        "answers with [Source N] citations",
        user_message="How long did the crew spend on the lunar surface, and who landed?",
        context=SAMPLE_CONTEXT,
        conversation_history=[],
    )

    check(
        "TEST 3 - answer absent from context",
        "declines instead of inventing a figure",
        user_message="What was the Apollo 11 mission budget in dollars?",
        context=SAMPLE_CONTEXT,
        conversation_history=[],
    )

    check(
        "TEST 4 - follow-up using conversation history",
        "resolves 'he' to Neil Armstrong from the history",
        user_message="What rank did he hold on the mission?",
        context=SAMPLE_CONTEXT,
        conversation_history=[
            {"role": "user", "content": "Who was the first person to step onto the Moon?"},
            {"role": "assistant", "content": "Neil Armstrong stepped onto the lunar surface first."},
        ],
    )

    print(f"\n{'=' * 70}\nAll 4 calls completed without error.")


if __name__ == "__main__":
    main()
