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
"""Weatherman persona — demo of ReActLoopEngine + WeatherToolBox + a local LLM.

This example wires together:

  - ``ReActLoopEngine`` (``ovos-agentic-loop``) as the agent loop.
  - ``OpenAIChatEngine`` (``ovos-openai-plugin``) pointing at a local
    OpenAI-compatible server (e.g. llama.cpp server, Ollama, LM Studio).
  - ``WeatherToolBox`` (``ovos-skill-weather``) for real-time weather via
    Open-Meteo (free, no API key required).

Run::

    python examples/weatherman_persona.py

Prerequisites::

    pip install ovos-agentic-loop ovos-openai-plugin ovos-skill-weather

    # Start a local OpenAI-compatible LLM server, e.g.:
    #   ollama serve
    #   ollama pull llama3

Environment variables (optional overrides):

    WEATHERMAN_MODEL   — model name to request (default: "llama3")
    WEATHERMAN_API_URL — base URL of the local LLM server
                         (default: "http://localhost:11434/v1")
    WEATHERMAN_LAT     — latitude for weather queries (default: 48.85 = Paris)
    WEATHERMAN_LON     — longitude for weather queries (default: 2.35)
    WEATHERMAN_TZ      — timezone (default: "Europe/Paris")
    WEATHERMAN_UNITS   — "metric" or "imperial" (default: "metric")
"""
import os
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Graceful import errors
# ---------------------------------------------------------------------------

_missing: list[str] = []
try:
    from ovos_agentic_loop.react import ReActLoopEngine
except ImportError:
    _missing.append("ovos-agentic-loop")

try:
    from ovos_openai_plugin.chat import OpenAIChatEngine  # type: ignore[import-untyped]
except ImportError:
    _missing.append("ovos-openai-plugin")

try:
    from ovos_skill_weather.weather_helpers.toolbox import WeatherToolBox  # type: ignore[import-untyped]
except ImportError:
    _missing.append("ovos-skill-weather")

if _missing:
    print("Missing dependencies — please install:")
    for pkg in _missing:
        print(f"  pip install {pkg}")
    sys.exit(1)

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = os.getenv("WEATHERMAN_MODEL", "llama3")
API_URL = os.getenv("WEATHERMAN_API_URL", "http://localhost:11434/v1")
LAT = float(os.getenv("WEATHERMAN_LAT", "48.8566"))
LON = float(os.getenv("WEATHERMAN_LON", "2.3522"))
TZ = os.getenv("WEATHERMAN_TZ", "Europe/Paris")
UNITS = os.getenv("WEATHERMAN_UNITS", "metric")

SYSTEM_PROMPT = (
    "You are a friendly, accurate weatherman assistant. "
    "When the user asks about the weather, use your weather tools to get "
    "real-time data before answering. "
    "Always mention the location and units in your reply. "
    "Be concise but informative — one or two short paragraphs is ideal."
)


# ---------------------------------------------------------------------------
# Persona builder
# ---------------------------------------------------------------------------

def build_weatherman() -> ReActLoopEngine:
    """Create and configure the weatherman persona.

    Returns:
        A ready-to-use :class:`ReActLoopEngine` wired with the local LLM
        and the weather toolbox.
    """
    # Brain: local LLM via OpenAI-compatible API.
    brain = OpenAIChatEngine(config={
        "model": MODEL,
        "api_url": API_URL,
        "api_key": "not-needed-for-local",  # most local servers ignore the key
        "temperature": 0.3,
        "max_tokens": 512,
    })

    # Weather toolbox: default location is set via the args below, but the
    # agent can override per-call if the user mentions a different city.
    weather_box = WeatherToolBox()

    # ReAct loop engine.
    agent = ReActLoopEngine(config={
        "max_iterations": 5,
        "system_prompt": SYSTEM_PROMPT,
    })
    agent.set_brain(brain)
    agent.load_toolboxes([weather_box])

    return agent


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def chat_loop(agent: ReActLoopEngine, default_lat: float = LAT,
              default_lon: float = LON, default_tz: str = TZ,
              default_units: str = UNITS) -> None:
    """Interactive REPL that injects default location context.

    The agent prompt prepends the user's coordinates so the LLM can call
    weather tools with the correct location when the user doesn't name a city.

    Args:
        agent: The configured :class:`ReActLoopEngine`.
        default_lat: Default latitude passed as context.
        default_lon: Default longitude passed as context.
        default_tz: Default timezone passed as context.
        default_units: Measurement units ("metric" or "imperial").
    """
    location_context = (
        f"[User location: lat={default_lat}, lon={default_lon}, "
        f"timezone={default_tz}, units={default_units}]"
    )
    history: list[AgentMessage] = []

    print("=" * 60)
    print("  WeatherMan Persona — powered by ReActLoopEngine")
    print(f"  Model   : {MODEL} @ {API_URL}")
    print(f"  Location: {default_lat}, {default_lon} ({default_tz})")
    print(f"  Units   : {default_units}")
    print("=" * 60)
    print("Type your weather questions.  'quit' or Ctrl-C to exit.\n")

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

        # Prepend location context so the LLM knows where to look.
        augmented_input = f"{location_context}\n\nUser: {user_input}"

        history.append(AgentMessage(role=MessageRole.USER, content=augmented_input))

        try:
            response = agent.continue_chat(history)
        except Exception as exc:
            print(f"[Error] {exc}")
            # Pop the failed user message so history stays clean.
            history.pop()
            continue

        print(f"\nWeatherMan: {response.content}\n")
        history.append(response)


# ---------------------------------------------------------------------------
# Persona config reference (JSON format for ovos-persona)
# ---------------------------------------------------------------------------

PERSONA_CONFIG = {
    "name": "WeatherMan",
    "solvers": ["ovos-react-loop"],
    "plugin-config": {
        "ovos-react-loop": {
            "brain": "ovos-chat-openai-plugin",
            "max_iterations": 5,
            "system_prompt": SYSTEM_PROMPT,
            "toolboxes": ["ovos-weather-tools"],
            "ovos-chat-openai-plugin": {
                "model": MODEL,
                "api_url": API_URL,
                "api_key": "not-needed-for-local",
                "temperature": 0.3,
            },
        }
    },
}
"""
Persona config dict for use with ovos-persona.

Load this persona via::

    from ovos_plugin_manager.persona import load_persona
    persona = load_persona("WeatherMan", PERSONA_CONFIG)
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    if "--print-config" in sys.argv:
        print(json.dumps(PERSONA_CONFIG, indent=2))
        sys.exit(0)

    try:
        weatherman = build_weatherman()
    except Exception as exc:
        print(f"[Startup error] {exc}")
        print("\nMake sure your local LLM server is running at:")
        print(f"  {API_URL}")
        sys.exit(1)

    chat_loop(weatherman)
