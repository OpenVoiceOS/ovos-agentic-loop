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
"""Regression tests for the ``tools`` kwarg contract.

``ovos_plugin_manager.templates.agents.ChatEngine.continue_chat`` declares
``tools`` unconditionally in its base signature. Callers such as
``ovos_persona_server/server_tools.py`` pass ``tools=`` by keyword to *any*
configured ChatEngine, including non-tool-capable ones. Every
``AgenticLoopEngine`` subclass must therefore accept (and may ignore) the
``tools`` kwarg or callers get a ``TypeError`` before the engine body ever
runs (Python validates the call signature before executing the function).

These engines are not tool-capable themselves (they don't set
``supports_tools = True``); this only proves the kwarg is *accepted*.
"""
from typing import List

import pytest

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

from ovos_agentic_loop.react import ReActLoopEngine
from ovos_agentic_loop.reflexion import ReflexionEngine
from ovos_agentic_loop.self_ask import SelfAskEngine
from ovos_agentic_loop.plan_execute import PlanAndExecuteEngine
from ovos_agentic_loop.chain_of_thought import ChainOfThoughtEngine
from ovos_agentic_loop.critic import CRITICEngine
from ovos_agentic_loop.tree_of_thoughts import TreeOfThoughtsEngine
from ovos_agentic_loop.native_toolcall import NativeToolCallEngine

ENGINE_CLASSES = [
    ReActLoopEngine,
    ReflexionEngine,
    SelfAskEngine,
    PlanAndExecuteEngine,
    ChainOfThoughtEngine,
    CRITICEngine,
    TreeOfThoughtsEngine,
    # already accepted `tools`, included to guard against regressions
    NativeToolCallEngine,
]


def _messages() -> List[AgentMessage]:
    return [AgentMessage(role=MessageRole.USER, content="hello")]


@pytest.mark.parametrize("engine_cls", ENGINE_CLASSES)
def test_continue_chat_accepts_tools_none(engine_cls) -> None:
    """`tools=None` (the base's default) must never raise TypeError."""
    engine = engine_cls()
    # No brain configured -> engines short-circuit with an error AgentMessage,
    # but only *after* argument binding succeeds, which is what we're testing.
    result = engine.continue_chat(_messages(), session_id="default",
                                   lang=None, units=None, tools=None)
    assert isinstance(result, AgentMessage)


@pytest.mark.parametrize("engine_cls", ENGINE_CLASSES)
def test_continue_chat_accepts_tools_list(engine_cls) -> None:
    """A real ``tools=[...]`` payload (as server_tools.py would pass) must
    also be accepted without raising, even though these engines ignore it."""
    engine = engine_cls()
    fake_tools = [{"type": "function", "function": {"name": "noop"}}]
    result = engine.continue_chat(_messages(), session_id="default",
                                   lang=None, units=None, tools=fake_tools)
    assert isinstance(result, AgentMessage)


@pytest.mark.parametrize("engine_cls", ENGINE_CLASSES)
def test_continue_chat_accepts_tools_as_keyword_only_call(engine_cls) -> None:
    """Mirrors ovos_persona_server/server_tools.py's call shape: everything
    after ``messages`` passed by keyword, including ``tools``."""
    engine = engine_cls()
    result = engine.continue_chat(
        messages=_messages(),
        session_id="default",
        lang=None,
        units=None,
        tools=[{"type": "function", "function": {"name": "noop"}}],
    )
    assert isinstance(result, AgentMessage)
