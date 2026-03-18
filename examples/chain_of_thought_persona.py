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
"""Chain-of-Thought persona — demo of ChainOfThoughtEngine + a local LLM.

This example wires together:

  - ``ChainOfThoughtEngine`` (``ovos-agentic-loop``) as the reasoning layer.
  - ``OpenAIChatEngine`` (``ovos-openai-plugin``) pointing at a local
    OpenAI-compatible server (e.g. Ollama, llama.cpp, LM Studio).

Chain-of-thought is the **simplest** agent pattern: a single LLM call with a
prompt that instructs the model to reason step-by-step before giving its
answer.  No tools, no iteration — just structured reasoning.

Best for: arithmetic, logic puzzles, multi-step instructions, common-sense
reasoning where no external data is needed.

Run::

    python examples/chain_of_thought_persona.py

Prerequisites::

    pip install ovos-agentic-loop ovos-openai-plugin

    # Start a local OpenAI-compatible LLM server, e.g.:
    #   ollama serve && ollama pull llama3

Environment variables (optional overrides):

    COT_MODEL      — model name to request (default: "llama3")
    COT_API_URL    — base URL of the local LLM server
                     (default: "http://localhost:11434/v1/chat/completions")
"""
import os
import sys
from typing import List

# ---------------------------------------------------------------------------
# Graceful import errors
# ---------------------------------------------------------------------------

_missing: list[str] = []
try:
    from ovos_agentic_loop.chain_of_thought import ChainOfThoughtEngine
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

MODEL = os.getenv("COT_MODEL", "llama3")
API_URL = os.getenv("COT_API_URL", "http://localhost:11434/v1/chat/completions")

SYSTEM_PROMPT = (
    "You are a careful, methodical reasoning assistant. "
    "For every question, think through the problem step by step before "
    "giving your final answer. "
    "Show your reasoning clearly — the user wants to follow your logic."
)

# ---------------------------------------------------------------------------
# Persona config reference (JSON format for ovos-persona)
# ---------------------------------------------------------------------------

PERSONA_CONFIG = {
    "name": "Reasoner",
    "solvers": ["ovos-chain-of-thought"],
    "ovos-chain-of-thought": {
        "brain": "ovos-chat-openai-plugin",
        "system_prompt": SYSTEM_PROMPT,
        "ovos-chat-openai-plugin": {
            "model": MODEL,
            "api_url": API_URL,
            "api_key": "not-needed-for-local",
            "temperature": 0.6,
            "max_tokens": 1024,
        },
    },
}
"""
Persona config dict for use with ovos-persona.

Load this persona via::

    from ovos_plugin_manager.persona import load_persona
    persona = load_persona("Reasoner", PERSONA_CONFIG)
"""


# ---------------------------------------------------------------------------
# Persona builder
# ---------------------------------------------------------------------------

def build_reasoner() -> "ChainOfThoughtEngine":
    """Create and configure the chain-of-thought persona.

    Returns:
        A ready-to-use :class:`ChainOfThoughtEngine` wired with the local LLM.
    """
    brain = OpenAIChatEngine(config={
        "model": MODEL,
        "api_url": API_URL,
        "api_key": "not-needed-for-local",
        "temperature": 0.6,
        "max_tokens": 1024,
    })

    agent = ChainOfThoughtEngine(config={"system_prompt": SYSTEM_PROMPT})
    agent.set_brain(brain)
    return agent


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def chat_loop(agent: "ChainOfThoughtEngine") -> None:
    """Interactive REPL for the chain-of-thought persona.

    Args:
        agent: The configured :class:`ChainOfThoughtEngine`.
    """
    history: List[AgentMessage] = []

    print("=" * 60)
    print("  Reasoner Persona — powered by ChainOfThoughtEngine")
    print(f"  Model : {MODEL} @ {API_URL}")
    print("=" * 60)
    print("Ask logic, math, or multi-step questions.  'quit' to exit.\n")

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

        print(f"\nReasoner: {response.content}\n")
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
        reasoner = build_reasoner()
    except Exception as exc:
        print(f"[Startup error] {exc}")
        print(f"\nMake sure your local LLM server is running at: {API_URL}")
        sys.exit(1)

    chat_loop(reasoner)
