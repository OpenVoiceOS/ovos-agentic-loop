#!/usr/bin/env python3
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
"""WikiGPT ReAct persona — multi-hop Wikipedia Q&A via ReActLoopEngine.

Architecture
------------
::

    User question
         │
         ▼
    ReActLoopEngine  ◄──── LLM (OpenAI-compatible)
         │
         ├─ search_wikipedia ──► Wikipedia Search API
         └─ get_wikipedia_page ──► Wikipedia Extract API

The LLM decides which tools to invoke, in which order, and how many times.
This enables multi-hop reasoning: e.g. "Who was the teacher of the man who
invented calculus?" can be answered by first searching for calculus, reading
the Newton article, then searching for Newton's teacher.

Usage
-----
::

    # Basic — use local LLM (Ollama default)
    python wikigpt_react_persona.py

    # Custom LLM endpoint
    WIKIGPT_API_URL=http://192.168.1.200:8000/v1/chat/completions \\
    WIKIGPT_MODEL=qwen3-8b \\
    python wikigpt_react_persona.py

    # Print resolved persona config and exit
    python wikigpt_react_persona.py --print-config

Environment variables
---------------------
WIKIGPT_API_URL : str
    Full ``/v1/chat/completions`` URL of the LLM endpoint.
    Default: ``http://localhost:11434/v1/chat/completions``
WIKIGPT_MODEL : str
    Model name forwarded to the LLM endpoint.
    Default: ``qwen3-8b``
WIKIGPT_MAX_ITER : int
    Maximum ReAct reasoning iterations before giving up.
    Default: ``6``
"""
import argparse
import json
import os
import sys

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_API_URL = os.environ.get(
    "WIKIGPT_API_URL", "http://localhost:11434/v1/chat/completions"
)
_MODEL = os.environ.get("WIKIGPT_MODEL", "qwen3-8b")
_MAX_ITER = int(os.environ.get("WIKIGPT_MAX_ITER", "6"))

PERSONA_CONFIG = {
    "name": "WikiGPT-ReAct",
    # ReActLoopEngine drives the reasoning loop
    "chat": "ovos-react-loop",
    "ovos-react-loop": {
        "max_iterations": _MAX_ITER,
        # LLM used inside the loop
        "llm": "ovos-openai-plugin",
        "ovos-openai-plugin": {
            "api_url": _API_URL,
            "model": _MODEL,
            "key": "nokey",
        },
        # Tools available to the LLM
        "toolboxes": ["ovos-wikipedia-tools"],
    },
}


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def _build_engine():
    """Construct and return a :class:`ReActLoopEngine` from ``PERSONA_CONFIG``."""
    try:
        from ovos_agentic_loop.engines.react import ReActLoopEngine
    except ImportError as exc:
        print(f"ERROR: ovos-agentic-loop not installed — {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        from ovos_wikipedia_solver.toolbox import WikipediaToolBox
    except ImportError as exc:
        print(f"ERROR: ovos-wikipedia-solver not installed — {exc}", file=sys.stderr)
        sys.exit(1)

    cfg = PERSONA_CONFIG["ovos-react-loop"]
    engine = ReActLoopEngine(cfg)
    # register the Wikipedia toolbox directly so no entry-point lookup is needed
    toolbox = WikipediaToolBox()
    for tool in toolbox.discover_tools():
        engine.register_tool(tool)
    return engine


def run_repl(engine) -> None:
    """Interactive REPL loop."""
    print("WikiGPT-ReAct  (type 'exit' or Ctrl-C to quit)")
    print(f"  LLM  : {_API_URL}  [{_MODEL}]")
    print(f"  Mode : ReAct loop (max {_MAX_ITER} iterations)")
    print()

    history = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Bye!")
            break

        history.append(AgentMessage(role=MessageRole.USER, content=user_input))
        try:
            response = engine.continue_chat(history)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            continue

        history.append(response)
        print(f"\nWikiGPT: {response.content}\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and launch the REPL or print config."""
    parser = argparse.ArgumentParser(
        description="WikiGPT-ReAct: multi-hop Wikipedia Q&A via ReActLoopEngine"
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved persona config JSON and exit.",
    )
    args = parser.parse_args()

    if args.print_config:
        print(json.dumps(PERSONA_CONFIG, indent=2))
        return

    engine = _build_engine()
    run_repl(engine)


if __name__ == "__main__":
    main()
