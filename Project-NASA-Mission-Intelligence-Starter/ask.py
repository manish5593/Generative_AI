#!/usr/bin/env python3
"""Ask the NASA assistant a one-off question from the command line.

Usage:
    python ask.py "What was Apollo 11?"
    python ask.py                          # uses the default question
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit("OPENAI_API_KEY not found. Create a .env file with:\n  OPENAI_API_KEY=\"...\"")

from llm_client import generate_response

question = " ".join(sys.argv[1:]) or "What was Apollo 11?"

print(generate_response(api_key, question, "", []))
