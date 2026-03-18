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
"""WikiGPT streaming demo — sentence-by-sentence output via ``stream_sentences()``.

Why streaming matters for voice assistants
-------------------------------------------
TTS engines work best when fed one sentence at a time: they can begin speaking
the first sentence while the solver is still computing the rest.  The
``ChatEngine.stream_sentences()`` method exposes exactly this interface.

``WikiGPTSolver.stream_sentences()`` splits the Wikipedia answer on newlines,
yielding each non-empty line as a separate "sentence".  This script shows three
output modes so you can observe the difference:

``--mode print``   (default)
    Print each sentence on its own line with a short delay to simulate TTS
    read-time.  Useful for testing response cadence in a terminal.

``--mode tts``
    Feed each sentence to an OPM TTS plugin via ``ovos_tts_plugin_mimic3``.
    Requires a TTS plugin installed in the current environment.

``--mode raw``
    Print the raw iterator output without any delay — useful for benchmarking.

Usage
-----
::

    # Interactive REPL with simulated TTS timing
    python wikigpt_streaming_persona.py

    # One-shot with raw sentence stream
    python wikigpt_streaming_persona.py --mode raw "Tell me about the Eiffel Tower"

    # Feed to a real TTS plugin (requires ovos-tts-plugin-mimic3 or similar)
    WIKIGPT_TTS_PLUGIN=ovos-tts-plugin-mimic3 \\
    python wikigpt_streaming_persona.py --mode tts

    # Print resolved config
    python wikigpt_streaming_persona.py --print-config

Environment variables
---------------------
WIKIGPT_ANSWER_MODE : str
    Answer mode passed to ``WikiGPTSolver``.  Default: ``FIRST_PARAGRAPH``.
WIKIGPT_LANG : str
    Wikipedia language code.  Default: ``en``.
WIKIGPT_TTS_PLUGIN : str
    OPM TTS plugin ID used in ``--mode tts``.  Default: ``None``.
WIKIGPT_WORDS_PER_MIN : int
    Simulated speaking rate for ``--mode print`` timing.  Default: ``150``.
"""
import argparse
import json
import os
import sys
import time

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ANSWER_MODE = os.environ.get("WIKIGPT_ANSWER_MODE", "FIRST_PARAGRAPH")
_LANG = os.environ.get("WIKIGPT_LANG", "en")
_TTS_PLUGIN = os.environ.get("WIKIGPT_TTS_PLUGIN") or None
_WPM = int(os.environ.get("WIKIGPT_WORDS_PER_MIN", "150"))

PERSONA_CONFIG = {
    "name": "WikiGPT-Streaming",
    "solvers": ["ovos-wikigpt"],
    "ovos-wikigpt": {
        "answer_mode": _ANSWER_MODE,
        "lang": _LANG,
        "max_pages": 3,
        "top_sections": 3,
    },
}


# ---------------------------------------------------------------------------
# Output handlers
# ---------------------------------------------------------------------------

def _speaking_delay(sentence: str) -> None:
    """Sleep for the approximate time it would take to speak ``sentence``."""
    word_count = len(sentence.split())
    delay = (word_count / _WPM) * 60
    time.sleep(max(0.3, delay))


def _output_print(sentence: str) -> None:
    """Print sentence then wait to simulate TTS read-time."""
    print(f"  > {sentence}")
    _speaking_delay(sentence)


def _output_tts(sentence: str, tts) -> None:
    """Speak sentence via an OPM TTS plugin instance."""
    try:
        audio_data = tts.get_tts(sentence, {})
        # audio_data is (wav_file, phonemes); play via aplay/afplay if available
        wav_file = audio_data[0] if isinstance(audio_data, (list, tuple)) else audio_data
        import subprocess
        subprocess.run(
            ["aplay", "-q", str(wav_file)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        # fall back to print on TTS error
        print(f"  > {sentence}  [TTS error: {exc}]")


def _output_raw(sentence: str) -> None:
    """Print sentence immediately without delay."""
    print(sentence)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _build_solver():
    """Construct and return a :class:`WikiGPTSolver`."""
    try:
        from ovos_wikipedia_solver.wikigpt import WikiGPTSolver
    except ImportError as exc:
        print(f"ERROR: ovos-wikipedia-solver not installed — {exc}", file=sys.stderr)
        sys.exit(1)
    return WikiGPTSolver(PERSONA_CONFIG["ovos-wikigpt"])


def _build_tts():
    """Load the configured TTS plugin, or return ``None``."""
    if not _TTS_PLUGIN:
        return None
    try:
        from ovos_plugin_manager.tts import OVOSTTSFactory
        return OVOSTTSFactory.create({"module": _TTS_PLUGIN})
    except Exception as exc:
        print(f"WARNING: could not load TTS plugin '{_TTS_PLUGIN}': {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def run_repl(solver, output_fn) -> None:
    """Interactive REPL — stream each answer sentence by sentence."""
    print(f"WikiGPT-Streaming  (mode={_ANSWER_MODE}, lang={_LANG})")
    print("Each sentence is yielded as soon as it is ready.")
    print("Type 'exit' to quit.\n")

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
        print("WikiGPT: ", end="", flush=True)
        if output_fn == _output_raw:
            print()  # newline before raw sentences

        sentences = []
        try:
            for sentence in solver.stream_sentences(history):
                output_fn(sentence)
                sentences.append(sentence)
        except Exception as exc:
            print(f"\nERROR: {exc}", file=sys.stderr)
            continue

        # reconstruct full response for history
        full = "\n".join(sentences)
        history.append(AgentMessage(role=MessageRole.ASSISTANT, content=full))
        print()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="WikiGPT streaming demo — sentence-by-sentence output"
    )
    parser.add_argument(
        "--mode",
        choices=["print", "tts", "raw"],
        default="print",
        help="Output mode: 'print' (with delay), 'tts' (OPM plugin), 'raw' (no delay).",
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
        print(json.dumps(PERSONA_CONFIG, indent=2))
        return

    solver = _build_solver()
    tts = _build_tts() if args.mode == "tts" else None

    if args.mode == "tts":
        if tts is None:
            print(
                "WARNING: TTS plugin unavailable — falling back to 'print' mode.",
                file=sys.stderr,
            )
            output_fn = _output_print
        else:
            output_fn = lambda s: _output_tts(s, tts)
    elif args.mode == "raw":
        output_fn = _output_raw
    else:
        output_fn = _output_print

    if args.question:
        msgs = [AgentMessage(role=MessageRole.USER, content=args.question)]
        for sentence in solver.stream_sentences(msgs):
            output_fn(sentence)
    else:
        run_repl(solver, output_fn)


if __name__ == "__main__":
    main()
