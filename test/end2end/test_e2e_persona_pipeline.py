"""Full-pipeline end-to-end test for ovos-agentic-loop via ovoscope.

Proves:
  1. An utterance flows through the real OVOS intent pipeline, hits the persona
     pipeline plugin, reaches the ovos-chain-of-thought-loop agent, and produces
     a ``speak`` message.
  2. Per-session memory is recorded by the live PersonaService.

No network access, no LLM downloads.  The loop's inner brain is monkeypatched
to return a deterministic fixed string, making CI fast and reproducible.

The ``ovos-chain-of-thought-loop`` is the simplest agentic loop: a single
inner LLM call whose full response is returned as the answer.  With the brain
monkeypatched, each turn completes in one step with no model inference.
"""
import json
import os
import tempfile

import pytest

import ovoscope

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session, SessionManager

from ovoscope import (
    PERSONA_PIPELINE,
    CaptureSession,
    get_minicroft,
    is_pipeline_available,
)

assert is_pipeline_available(PERSONA_PIPELINE), (
    "ovos-persona-pipeline-plugin must be installed (ships with ovos-persona)"
)

# ---------------------------------------------------------------------------
# Deterministic brain monkeypatch
# ---------------------------------------------------------------------------
# Installed before the persona service loads so any ChainOfThoughtEngine
# instantiated during the test uses the fake brain.

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_agentic_loop.chain_of_thought import ChainOfThoughtEngine

_FIXED_REPLY = "I am the Looper agent and I am ready to help."


class _FakeBrain:
    """Deterministic stand-in for an inner LLM — always returns a fixed string."""

    def continue_chat(self, messages, session_id="default", lang=None, units=None):
        return AgentMessage(role=MessageRole.ASSISTANT, content=_FIXED_REPLY)


def _patched_load_brain(self):
    return _FakeBrain()


_original_load_brain = ChainOfThoughtEngine._load_brain


@pytest.fixture(scope="module", autouse=True)
def _patch_chain_of_thought_brain():
    """Patch ChainOfThoughtEngine._load_brain for the duration of this module only."""
    ChainOfThoughtEngine._load_brain = _patched_load_brain
    yield
    ChainOfThoughtEngine._load_brain = _original_load_brain

# ---------------------------------------------------------------------------
# Persona JSON setup
# ---------------------------------------------------------------------------

PERSONA_NAME = "Looper"
LOOP_PLUGIN = "ovos-chain-of-thought-loop"


def _make_personas_dir() -> str:
    """Write a minimal Looper persona JSON into a temp directory."""
    tmpdir = tempfile.mkdtemp()
    persona = {
        "name": PERSONA_NAME,
        "handlers": [LOOP_PLUGIN],
        # Per-loop config block: no brain key needed because _load_brain is patched.
        LOOP_PLUGIN: {},
    }
    with open(os.path.join(tmpdir, f"{PERSONA_NAME}.json"), "w") as fh:
        json.dump(persona, fh)
    return tmpdir


PERSONAS_PATH = _make_personas_dir()

PIPELINE_CONFIG = {
    "persona": {
        "personas_path": PERSONAS_PATH,
        "default_persona": PERSONA_NAME,
        "short-term-memory": True,
        "handle_fallback": True,
        "ignore_plugin_personas": True,
    }
}

TEST_PIPELINE = [
    "ovos-persona-pipeline-plugin-high",
    "ovos-persona-pipeline-plugin-low",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utterance_msg(utterance: str, sess: Session) -> Message:
    return Message(
        "recognizer_loop:utterance",
        {"utterances": [utterance], "lang": sess.lang},
        {"session": sess.serialize()},
    )


def _drive_utterance(croft, sess: Session, utterance: str, timeout: int = 30):
    cap = CaptureSession(
        croft,
        eof_msgs=["ovos.utterance.handled", "ovos.utterance.cancelled"],
    )
    cap.capture(_utterance_msg(utterance, sess), timeout=timeout)
    return cap.finish()


def _get_persona_service(croft):
    return croft.intents.pipeline_plugins["ovos-persona-pipeline-plugin"]


# ---------------------------------------------------------------------------
# Module-level MiniCroft (shared for speed)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mc():
    croft = get_minicroft(
        skill_ids=[],
        default_pipeline=TEST_PIPELINE,
        pipeline_config=PIPELINE_CONFIG,
        max_wait=180,
    )
    yield croft
    croft.stop()


# ---------------------------------------------------------------------------
# Test 1: loop persona produces a speak through the full pipeline
# ---------------------------------------------------------------------------


class TestLooperPersonaSpeaks:
    """The utterance must traverse the full OVOS intent pipeline, reach the
    chain-of-thought loop agent, and produce a non-empty ``speak`` message."""

    def test_pipeline_produces_speak(self, mc):
        sess = Session(session_id="loop-e2e-speak-1")
        SessionManager.sessions[sess.session_id] = sess

        messages = _drive_utterance(mc, sess, "hello there", timeout=30)

        msg_types = [m.msg_type for m in messages]
        speak_msgs = [m for m in messages if m.msg_type == "speak"]

        assert speak_msgs, (
            f"Expected at least one 'speak' message; got msg_types: {msg_types}"
        )
        spoken = speak_msgs[0].data.get("utterance", "")
        assert spoken.strip(), (
            f"'speak' message had empty utterance; data={speak_msgs[0].data}"
        )

    def test_speak_contains_loop_reply(self, mc):
        """The spoken text must come from the loop agent (our fixed reply)."""
        sess = Session(session_id="loop-e2e-speak-2")
        SessionManager.sessions[sess.session_id] = sess

        messages = _drive_utterance(mc, sess, "what can you do", timeout=30)

        speak_msgs = [m for m in messages if m.msg_type == "speak"]
        assert speak_msgs, "No speak message produced by the loop persona"

        spoken = speak_msgs[0].data.get("utterance", "")
        assert spoken.strip(), f"speak message is empty; data={speak_msgs[0].data}"


# ---------------------------------------------------------------------------
# Test 2: per-session memory is recorded
# ---------------------------------------------------------------------------


class TestLooperMemory:
    """PersonaService records USER + ASSISTANT turns per session_id when
    short-term memory is enabled."""

    def test_user_turn_recorded_in_memory(self, mc):
        svc = _get_persona_service(mc)
        sess = Session(session_id="loop-e2e-mem-user")
        SessionManager.sessions[sess.session_id] = sess

        persona = svc.personas.get(PERSONA_NAME)
        assert persona is not None, f"Persona '{PERSONA_NAME}' not loaded"
        assert persona.memory is not None, "Persona must have short-term memory enabled"

        _drive_utterance(mc, sess, "remember this for me", timeout=30)

        history = persona.memory.get_history(sess.session_id)
        contents = [m.content for m in history]
        assert any("remember this for me" in c for c in contents), (
            f"User utterance not found in memory for session {sess.session_id}. "
            f"History: {contents}"
        )

    def test_assistant_response_recorded_in_memory(self, mc):
        svc = _get_persona_service(mc)
        sess = Session(session_id="loop-e2e-mem-assistant")
        SessionManager.sessions[sess.session_id] = sess

        persona = svc.personas.get(PERSONA_NAME)
        assert persona is not None
        assert persona.memory is not None

        _drive_utterance(mc, sess, "say something back", timeout=30)

        history = persona.memory.get_history(sess.session_id)
        roles = [m.role for m in history]
        assert MessageRole.ASSISTANT in roles, (
            f"No ASSISTANT turn recorded in memory. History roles: {roles}"
        )

    def test_unknown_session_has_empty_history(self, mc):
        svc = _get_persona_service(mc)
        persona = svc.personas.get(PERSONA_NAME)
        assert persona is not None
        assert persona.memory is not None

        sess = Session(session_id="loop-e2e-mem-known")
        SessionManager.sessions[sess.session_id] = sess
        _drive_utterance(mc, sess, "hello", timeout=30)

        unknown_history = persona.memory.get_history("session-that-never-existed-loop")
        assert unknown_history == [], (
            f"Expected empty history for unknown session, got: {unknown_history}"
        )

    def test_same_session_accumulates_turns(self, mc):
        """Two utterances on the same session must produce at least 2 history entries."""
        svc = _get_persona_service(mc)
        sess = Session(session_id="loop-e2e-mem-accumulate")
        SessionManager.sessions[sess.session_id] = sess

        persona = svc.personas.get(PERSONA_NAME)
        assert persona is not None
        assert persona.memory is not None
        persona.memory.session2history.pop(sess.session_id, None)

        _drive_utterance(mc, sess, "first question", timeout=30)
        _drive_utterance(mc, sess, "second question", timeout=30)

        history = persona.memory.get_history(sess.session_id)
        assert len(history) >= 2, (
            f"Expected at least 2 history entries after two turns, got {len(history)}: "
            f"{[m.content for m in history]}"
        )
