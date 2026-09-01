from typing import Dict, List

import httpx
from openai import OpenAI

# The default 5s connect timeout is too tight for the Vocareum proxy, which can
# take ~15s to establish a connection.
REQUEST_TIMEOUT = httpx.Timeout(120.0, connect=60.0)

# How many previous turns (user + assistant messages) to carry into the request.
MAX_HISTORY_MESSAGES = 10

SYSTEM_PROMPT = """You are a NASA mission intelligence assistant with deep expertise in \
human spaceflight programs, including Mercury, Gemini, Apollo, Skylab, the Space Shuttle, \
and the International Space Station.

Answer questions using the retrieved mission documents provided to you:
- Ground every factual claim in the provided context, and cite the source number (e.g. "[Source 2]") \
when you use it.
- If the context does not contain the answer, say so plainly and explain what is missing rather \
than guessing. Only fall back on general knowledge when you clearly label it as such.
- Preserve technical precision: keep mission designations, dates, altitudes, and measurements \
exactly as they appear in the source material.
- Be concise and factual. Prefer short paragraphs or bullet points over long narrative."""


def generate_response(openai_key: str, user_message: str, context: str,
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # Define system prompt
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Set context in messages
    if context:
        messages.append({
            "role": "system",
            "content": f"Retrieved NASA mission documents for this question:\n\n{context}"
        })
    else:
        messages.append({
            "role": "system",
            "content": "No mission documents were retrieved for this question. Tell the user "
                       "you could not find supporting documents before answering."
        })

    # Add chat history (most recent turns only, normalized to role/content)
    for message in (conversation_history or [])[-MAX_HISTORY_MESSAGES:]:
        role = message.get("role")
        content = message.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    # Current question
    messages.append({"role": "user", "content": user_message})

    # Create OpenAI Client
    client = OpenAI(api_key=openai_key, timeout=REQUEST_TIMEOUT)

    # Send request to OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,  # low temperature keeps answers close to the source documents
        max_tokens=800,
    )

    # Return response
    return response.choices[0].message.content.strip()
