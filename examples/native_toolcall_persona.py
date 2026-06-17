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
"""Native tool-calling demo — NativeToolCallEngine + MathToolBox + a local LLM.

Unlike the ReAct examples, this drives the brain's *native* function-calling: the
toolboxes are handed to the brain via ``continue_chat(tools=...)`` and structured
``tool_calls`` come back, so it needs a brain whose ChatEngine sets
``supports_tools = True`` (e.g. ``ovos-openai-plugin`` against a tool-capable model).
If the brain lacks native tool support, NativeToolCallEngine falls back to ReAct.

Run::

    python examples/native_toolcall_persona.py

Prerequisites::

    pip install ovos-agentic-loop ovos-openai-plugin
    # a local OpenAI-compatible server with a tool-capable model, e.g.:
    #   ollama serve && ollama pull qwen2.5

Environment overrides: CALC_MODEL, CALC_API_URL.
"""
import os
import sys

_missing = []
try:
    from ovos_agentic_loop.native_toolcall import NativeToolCallEngine
except ImportError:
    _missing.append("ovos-agentic-loop")
try:
    from ovos_openai_plugin.chat import OpenAIChatEngine  # type: ignore[import-untyped]
except ImportError:
    _missing.append("ovos-openai-plugin")
try:
    from ovos_agentic_loop.tools.math import MathToolBox
except ImportError:
    _missing.append("ovos-agentic-loop[math]")

if _missing:
    sys.exit(f"Missing dependencies: {', '.join(_missing)}")


def main() -> None:
    brain = OpenAIChatEngine({
        "model": os.environ.get("CALC_MODEL", "qwen2.5"),
        "api_url": os.environ.get("CALC_API_URL",
                                  "http://localhost:11434/v1/chat/completions"),
        "api_key": "not-needed-for-local",
        "system_prompt": "You are a precise assistant. Use the math tools to "
                         "compute answers instead of guessing.",
    })
    if not getattr(brain, "supports_tools", False):
        print("NOTE: this brain reports no native tool support — the engine will "
              "fall back to the ReAct text loop.")

    engine = NativeToolCallEngine({"max_iterations": 5})
    engine.set_brain(brain)
    engine.load_toolboxes([MathToolBox(toolbox_id="ovos-math-tools")])

    print("Calculator agent — ask a math question (Ctrl-C to quit).\n")
    try:
        while True:
            query = input("you> ").strip()
            if not query:
                continue
            answer = engine.get_response(query)
            print(f"bot> {answer}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nbye!")


if __name__ == "__main__":
    main()
