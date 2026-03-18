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
"""WikiGPT multi-language demo — automatic language detection via ``lang="auto"``.

How ``lang="auto"`` works
-------------------------
1. The question is passed to an OPM language-detector plugin configured via the
   ``lang_detector`` key.
2. The detected BCP-47 code (e.g. ``"de"``, ``"pt"``) is used to select the
   matching Wikipedia edition.
3. If detection fails or no plugin is configured the solver falls back to
   ``"en"``.

This script works **without** a language-detector plugin by letting you pass
``--lang`` explicitly.  When ``--lang auto`` is set and no detector is
configured the fallback to English is logged, so you can still observe the
path through the code.

Usage
-----
::

    # Auto-detect language (requires lang_detector plugin, else falls back to en)
    python wikigpt_multilang_persona.py --lang auto

    # Force German Wikipedia
    python wikigpt_multilang_persona.py --lang de

    # Run a one-shot question non-interactively
    python wikigpt_multilang_persona.py --lang pt "Quem foi Isaac Newton?"

    # Print resolved persona config
    python wikigpt_multilang_persona.py --print-config

Environment variables
---------------------
WIKIGPT_ANSWER_MODE : str
    Answer mode.  Default: ``FIRST_PARAGRAPH`` (no LLM required).
    Set to ``RAG`` and configure WIKIGPT_LLM_* for LLM-backed answers.
WIKIGPT_LLM_URL : str
    LLM endpoint URL (only used when WIKIGPT_ANSWER_MODE=RAG).
    Default: ``http://localhost:11434/v1/chat/completions``
WIKIGPT_LLM_MODEL : str
    LLM model name (only used when WIKIGPT_ANSWER_MODE=RAG).
    Default: ``qwen3-8b``
WIKIGPT_LANG_DETECTOR : str
    OPM language-detector plugin ID.  Default: ``None`` (auto falls back to en).
"""
import argparse
import json
import os
import sys

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ANSWER_MODE = os.environ.get("WIKIGPT_ANSWER_MODE", "FIRST_PARAGRAPH")
_LLM_URL = os.environ.get("WIKIGPT_LLM_URL", "http://localhost:11434/v1/chat/completions")
_LLM_MODEL = os.environ.get("WIKIGPT_LLM_MODEL", "qwen3-8b")
_LANG_DETECTOR = os.environ.get("WIKIGPT_LANG_DETECTOR") or None


def _build_persona_config(lang: str) -> dict:
    """Build the WikiGPT persona config for the given lang setting."""
    solver_cfg: dict = {
        "answer_mode": _ANSWER_MODE,
        "lang": lang,
        "max_pages": 3,
        "top_sections": 3,
    }
    if _LANG_DETECTOR:
        solver_cfg["lang_detector"] = _LANG_DETECTOR
    if _ANSWER_MODE == "RAG":
        solver_cfg["llm"] = "ovos-openai-plugin"
        solver_cfg["ovos-openai-plugin"] = {
            "api_url": _LLM_URL,
            "model": _LLM_MODEL,
            "key": "nokey",
        }
    return {
        "name": "WikiGPT-MultiLang",
        "solvers": ["ovos-wikigpt"],
        "ovos-wikigpt": solver_cfg,
    }


# ---------------------------------------------------------------------------
# Engine builder
# ---------------------------------------------------------------------------

def _build_solver(lang: str):
    """Construct and return a :class:`WikiGPTSolver` for the given lang."""
    try:
        from ovos_wikipedia_solver.wikigpt import WikiGPTSolver
    except ImportError as exc:
        print(f"ERROR: ovos-wikipedia-solver not installed — {exc}", file=sys.stderr)
        sys.exit(1)

    cfg = _build_persona_config(lang)["ovos-wikigpt"]
    return WikiGPTSolver(cfg)


# ---------------------------------------------------------------------------
# REPL / one-shot
# ---------------------------------------------------------------------------

def run_repl(solver, lang: str) -> None:
    """Interactive REPL loop."""
    lang_display = f"auto (detector: {_LANG_DETECTOR})" if lang == "auto" else lang
    print(f"WikiGPT-MultiLang  (lang={lang_display}, mode={_ANSWER_MODE})")
    print("Type a question in any language.  Type 'exit' to quit.\n")

    history = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Bye!")
            break

        history.append(AgentMessage(role=MessageRole.USER, content=user_input))
        try:
            # pass lang=None so the solver uses its configured default / auto-detect
            response = solver.continue_chat(history)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            continue

        history.append(response)
        print(f"\nWikiGPT: {response.content}\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="WikiGPT multi-language demo (lang='auto' or explicit code)"
    )
    parser.add_argument(
        "--lang",
        default="auto",
        help="Wikipedia language code ('auto' for detection, or e.g. 'de', 'pt'). Default: auto",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved persona config JSON and exit.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="One-shot question (non-interactive mode).",
    )
    args = parser.parse_args()

    if args.print_config:
        print(json.dumps(_build_persona_config(args.lang), indent=2))
        return

    solver = _build_solver(args.lang)

    if args.question:
        # one-shot mode
        msgs = [AgentMessage(role=MessageRole.USER, content=args.question)]
        response = solver.continue_chat(msgs)
        print(response.content)
    else:
        run_repl(solver, args.lang)


if __name__ == "__main__":
    main()
