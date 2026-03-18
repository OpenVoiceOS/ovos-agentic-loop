#!/usr/bin/env python3
# Copyright 2025, OpenVoiceOS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reflexion persona — ReflexionEngine + WebSearch + local LLM.

This example wires together:

  - ``ReflexionEngine`` (``ovos-agentic-loop``) as the self-correcting loop.
  - ``OpenAIChatEngine`` (``ovos-openai-plugin``) pointing at a local
    OpenAI-compatible server.
  - ``WebSearchToolBox`` (bundled with ``ovos-agentic-loop``) for fact-checking.

Reflexion wraps a standard ReAct loop with verbal self-reflection: after each
episode the agent evaluates its own answer, generates a critique if it fell
short, and uses that critique as a "lesson learned" in the next episode.  This
helps catch reasoning errors and hallucinations without any external reward
signal.

Best for: tasks where the first attempt is likely wrong and self-correction
is valuable (e.g. complex multi-hop questions, code generation, planning).

Run::

    python examples/reflexion_persona.py

Prerequisites::

    pip install "ovos-agentic-loop[web]" ovos-openai-plugin

    # Start a local OpenAI-compatible LLM server, e.g.:
    #   ollama serve && ollama pull llama3

Environment variables (optional overrides):

    REFLEXION_MODEL        — model name to request (default: "llama3")
    REFLEXION_API_URL      — base URL of the local LLM server
                             (default: "http://localhost:11434/v1/chat/completions")
    REFLEXION_MAX_ROUNDS   — max self-reflection episodes (default: 3)
"""
import os
import sys
from typing import List

# ---------------------------------------------------------------------------
# Graceful import errors
# ---------------------------------------------------------------------------

_missing: list[str] = []
try:
    from ovos_agentic_loop.reflexion import ReflexionEngine
    from ovos_agentic_loop.tools.web import WebSearchToolBox
except ImportError:
    _missing.append("ovos-agentic-loop")

try:
    from ovos_openai_plugin.chat import OpenAIChatEngine  # type: ignore[import-untyped]
except ImportError:
    _missing.append("ovos-openai-plugin")

if _missing:
    print("Missing dependencies — please install:")
    for pkg in _missing:
        print(f"  pip install {pkg}")
    sys.exit(1)

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = os.getenv("REFLEXION_MODEL", "llama3")
API_URL = os.getenv("REFLEXION_API_URL", "http://localhost:11434/v1/chat/completions")
MAX_ROUNDS = int(os.getenv("REFLEXION_MAX_ROUNDS", "3"))

SYSTEM_PROMPT = (
    "You are a precise, self-critical assistant. "
    "Think carefully before answering. "
    "If you are unsure about a fact, search the web to verify it. "
    "Accuracy matters more than speed."
)

# ---------------------------------------------------------------------------
# Persona config reference (JSON format for ovos-persona)
# ---------------------------------------------------------------------------

PERSONA_CONFIG = {
    "name": "Reflexion",
    "solvers": ["ovos-reflexion-loop"],
    "ovos-reflexion-loop": {
        "brain": "ovos-chat-openai-plugin",
        "max_reflections": MAX_ROUNDS,
        "system_prompt": SYSTEM_PROMPT,
        "toolboxes": ["ovos-web-search-tools"],
        "ovos-chat-openai-plugin": {
            "model": MODEL,
            "api_url": API_URL,
            "api_key": "not-needed-for-local",
            "temperature": 0.4,
            "max_tokens": 1024,
        },
    },
}
"""
Persona config dict for use with ovos-persona.

Load this persona via::

    from ovos_plugin_manager.persona import load_persona
    persona = load_persona("Reflexion", PERSONA_CONFIG)
"""


# ---------------------------------------------------------------------------
# Persona builder
# ---------------------------------------------------------------------------

def build_reflexion() -> "ReflexionEngine":
    """Create and configure the reflexion persona.

    Returns:
        A ready-to-use :class:`ReflexionEngine` wired with web search.
    """
    brain = OpenAIChatEngine(config={
        "model": MODEL,
        "api_url": API_URL,
        "api_key": "not-needed-for-local",
        "temperature": 0.4,
        "max_tokens": 1024,
    })

    web_search = WebSearchToolBox()

    agent = ReflexionEngine(config={
        "max_reflections": MAX_ROUNDS,
        "system_prompt": SYSTEM_PROMPT,
    })
    agent.set_brain(brain)
    agent.load_toolboxes([web_search])

    return agent


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def chat_loop(agent: "ReflexionEngine") -> None:
    """Interactive REPL for the reflexion persona.

    Args:
        agent: The configured :class:`ReflexionEngine`.
    """
    history: List[AgentMessage] = []

    print("=" * 60)
    print("  Reflexion Persona — powered by ReflexionEngine")
    print(f"  Model         : {MODEL} @ {API_URL}")
    print(f"  Max episodes  : {MAX_ROUNDS}")
    print("=" * 60)
    print("The agent will self-critique and retry if its answer is weak.")
    print("Type your questions.  'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        history.append(AgentMessage(role=MessageRole.USER, content=user_input))

        try:
            response = agent.continue_chat(history)
        except Exception as exc:
            print(f"[Error] {exc}")
            history.pop()
            continue

        print(f"\nReflexion: {response.content}\n")
        history.append(response)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    if "--print-config" in sys.argv:
        print(json.dumps(PERSONA_CONFIG, indent=2))
        sys.exit(0)

    try:
        agent = build_reflexion()
    except Exception as exc:
        print(f"[Startup error] {exc}")
        print(f"\nMake sure your local LLM server is running at: {API_URL}")
        sys.exit(1)

    chat_loop(agent)
