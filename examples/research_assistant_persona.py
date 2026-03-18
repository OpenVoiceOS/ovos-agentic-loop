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
"""Research assistant persona — ReActLoopEngine + WebSearch + Clock tools + local LLM.

This example wires together:

  - ``ReActLoopEngine`` (``ovos-agentic-loop``) as the agent loop.
  - ``OpenAIChatEngine`` (``ovos-openai-plugin``) pointing at a local
    OpenAI-compatible server.
  - ``WebSearchToolBox`` (bundled with ``ovos-agentic-loop``) for live
    DuckDuckGo searches.
  - ``ClockToolBox`` (bundled with ``ovos-agentic-loop``) for current
    date/time awareness.

The agent can look up current events, verify facts, and answer questions that
require real-time information.

Run::

    python examples/research_assistant_persona.py

Prerequisites::

    pip install "ovos-agentic-loop[web]" ovos-openai-plugin

    # Start a local OpenAI-compatible LLM server, e.g.:
    #   ollama serve && ollama pull llama3

Environment variables (optional overrides):

    RESEARCH_MODEL    — model name to request (default: "llama3")
    RESEARCH_API_URL  — base URL of the local LLM server
                        (default: "http://localhost:11434/v1/chat/completions")
    RESEARCH_RESULTS  — max web search results per query (default: 5)
"""
import os
import sys
from typing import List

# ---------------------------------------------------------------------------
# Graceful import errors
# ---------------------------------------------------------------------------

_missing: list[str] = []
try:
    from ovos_agentic_loop.react import ReActLoopEngine
    from ovos_agentic_loop.tools.web import WebSearchToolBox
    from ovos_agentic_loop.tools.clock import ClockToolBox
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

MODEL = os.getenv("RESEARCH_MODEL", "llama3")
API_URL = os.getenv("RESEARCH_API_URL", "http://localhost:11434/v1/chat/completions")
MAX_RESULTS = int(os.getenv("RESEARCH_RESULTS", "5"))

SYSTEM_PROMPT = (
    "You are a thorough research assistant with access to web search and a clock. "
    "When answering questions about current events, recent facts, or anything that "
    "might have changed recently, always search the web first. "
    "When the user asks about time or today's date, use the clock tool. "
    "Cite your sources when possible. Be accurate and concise."
)

# ---------------------------------------------------------------------------
# Persona config reference (JSON format for ovos-persona)
# ---------------------------------------------------------------------------

PERSONA_CONFIG = {
    "name": "Researcher",
    "solvers": ["ovos-react-loop"],
    "ovos-react-loop": {
        "brain": "ovos-chat-openai-plugin",
        "max_iterations": 6,
        "system_prompt": SYSTEM_PROMPT,
        "toolboxes": ["ovos-web-search-tools", "ovos-clock-tools"],
        "ovos-chat-openai-plugin": {
            "model": MODEL,
            "api_url": API_URL,
            "api_key": "not-needed-for-local",
            "temperature": 0.3,
            "max_tokens": 1024,
        },
    },
}
"""
Persona config dict for use with ovos-persona.

Load this persona via::

    from ovos_plugin_manager.persona import load_persona
    persona = load_persona("Researcher", PERSONA_CONFIG)
"""


# ---------------------------------------------------------------------------
# Persona builder
# ---------------------------------------------------------------------------

def build_researcher() -> "ReActLoopEngine":
    """Create and configure the research assistant persona.

    Returns:
        A ready-to-use :class:`ReActLoopEngine` wired with web search and clock.
    """
    brain = OpenAIChatEngine(config={
        "model": MODEL,
        "api_url": API_URL,
        "api_key": "not-needed-for-local",
        "temperature": 0.3,
        "max_tokens": 1024,
    })

    web_search = WebSearchToolBox(config={"max_results": MAX_RESULTS})
    clock = ClockToolBox()

    agent = ReActLoopEngine(config={
        "max_iterations": 6,
        "system_prompt": SYSTEM_PROMPT,
    })
    agent.set_brain(brain)
    agent.load_toolboxes([web_search, clock])

    return agent


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def chat_loop(agent: "ReActLoopEngine") -> None:
    """Interactive REPL for the research assistant persona.

    Args:
        agent: The configured :class:`ReActLoopEngine`.
    """
    history: List[AgentMessage] = []

    print("=" * 60)
    print("  Researcher Persona — powered by ReActLoopEngine")
    print(f"  Model   : {MODEL} @ {API_URL}")
    print(f"  Tools   : web_search (max {MAX_RESULTS} results), get_current_datetime")
    print("=" * 60)
    print("Ask anything that requires current information.  'quit' to exit.\n")

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

        print(f"\nResearcher: {response.content}\n")
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
        researcher = build_researcher()
    except Exception as exc:
        print(f"[Startup error] {exc}")
        print(f"\nMake sure your local LLM server is running at: {API_URL}")
        sys.exit(1)

    chat_loop(researcher)
