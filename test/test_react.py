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

"""Unit tests for ReActLoopEngine."""
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_agentic_loop.react import (
    ReActLoopEngine,
    _FINAL_ANSWER_TOKEN,
    _build_react_system,
    _extract_final_answer,
    _parse_action,
)


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------

class TestParseAction:
    def test_valid_action(self) -> None:
        text = (
            "Thought: I should search.\n"
            "Action: web_search\n"
            'Action Input: {"query": "OpenVoiceOS"}'
        )
        result = _parse_action(text)
        assert result is not None
        name, args = result
        assert name == "web_search"
        assert args == {"query": "OpenVoiceOS"}

    def test_missing_action_returns_none(self) -> None:
        assert _parse_action("Just a thought.") is None

    def test_invalid_json_returns_none(self) -> None:
        text = "Action: foo\nAction Input: {bad json}"
        assert _parse_action(text) is None


class TestExtractFinalAnswer:
    def test_extracts_answer(self) -> None:
        text = f"Thought: done.\n{_FINAL_ANSWER_TOKEN} Paris"
        assert _extract_final_answer(text) == "Paris"

    def test_missing_token_returns_none(self) -> None:
        assert _extract_final_answer("some text") is None


class TestBuildReactSystem:
    def test_contains_tool_schemas(self) -> None:
        schemas = [{"name": "calc", "description": "math"}]
        prompt = _build_react_system(schemas)
        assert "calc" in prompt
        assert _FINAL_ANSWER_TOKEN in prompt


# ---------------------------------------------------------------------------
# ReActLoopEngine behaviour tests (brain mocked)
# ---------------------------------------------------------------------------

def _make_brain(responses: List[str]) -> MagicMock:
    """Create a mock ChatEngine that returns canned text responses in order."""
    brain = MagicMock()
    brain.continue_chat.side_effect = [
        AgentMessage(role=MessageRole.ASSISTANT, content=r)
        for r in responses
    ]
    return brain


class TestReActLoopEngineFinalAnswer:
    def test_immediate_final_answer(self) -> None:
        engine = ReActLoopEngine()
        engine.set_brain(_make_brain([f"Thought: I know.\n{_FINAL_ANSWER_TOKEN} 42"]))
        msg = AgentMessage(role=MessageRole.USER, content="What is 6*7?")
        result = engine.continue_chat([msg])
        assert result.role == MessageRole.ASSISTANT
        assert result.content == "42"

    def test_no_brain_returns_error(self) -> None:
        engine = ReActLoopEngine()
        msg = AgentMessage(role=MessageRole.USER, content="hi")
        result = engine.continue_chat([msg])
        assert "Error" in result.content


class TestReActLoopEngineToolCall:
    def test_single_tool_then_final_answer(self) -> None:
        engine = ReActLoopEngine()

        # First turn: action; second turn: final answer.
        brain = _make_brain([
            'Thought: use calc\nAction: calc\nAction Input: {"a": 1, "b": 2}',
            f"Thought: got 3.\n{_FINAL_ANSWER_TOKEN} 3",
        ])
        engine.set_brain(brain)

        # Mock toolbox with a "calc" tool.
        tb = MagicMock()
        tb.tool_json_list = [{"name": "calc"}]
        tb.get_tool.return_value = MagicMock()
        tb.call_tool.return_value = "3"
        engine.load_toolboxes([tb])

        msg = AgentMessage(role=MessageRole.USER, content="1+2?")
        result = engine.continue_chat([msg])
        assert result.content == "3"
        tb.call_tool.assert_called_once_with("calc", {"a": 1, "b": 2})

    def test_unknown_tool_returns_error_observation(self) -> None:
        engine = ReActLoopEngine()
        brain = _make_brain([
            'Thought: try\nAction: missing_tool\nAction Input: {"x": 1}',
            f"{_FINAL_ANSWER_TOKEN} done",
        ])
        engine.set_brain(brain)

        tb = MagicMock()
        tb.tool_json_list = []
        tb.get_tool.return_value = None
        engine.load_toolboxes([tb])

        msg = AgentMessage(role=MessageRole.USER, content="go")
        result = engine.continue_chat([msg])
        # Loop still completes — the observation error is forwarded to the brain.
        assert result.role == MessageRole.ASSISTANT


class TestReActLoopEngineMaxIterations:
    def test_max_iterations_fallback(self) -> None:
        engine = ReActLoopEngine(config={"max_iterations": 2})

        # Always return an action, never a final answer, to exhaust iterations.
        endless = ['Thought: try\nAction: foo\nAction Input: {"x": 1}'] * 3
        endless.append(f"{_FINAL_ANSWER_TOKEN} gave up")
        brain = _make_brain(endless)
        engine.set_brain(brain)

        tb = MagicMock()
        tb.tool_json_list = [{"name": "foo"}]
        tb.get_tool.return_value = MagicMock()
        tb.call_tool.return_value = "result"
        engine.load_toolboxes([tb])

        msg = AgentMessage(role=MessageRole.USER, content="loop forever")
        result = engine.continue_chat([msg])
        assert result.role == MessageRole.ASSISTANT
        # Brain called at most max_iterations + 1 times.
        assert brain.continue_chat.call_count <= engine.max_iterations + 2


class TestReActNoActionParsed:
    def test_response_without_action_or_final_answer(self) -> None:
        """When the LLM returns plain text (no action, no FINAL_ANSWER), treat it as final."""
        engine = ReActLoopEngine()
        engine.set_brain(_make_brain(["Just some plain text response."]))
        msg = AgentMessage(role=MessageRole.USER, content="tell me something")
        result = engine.continue_chat([msg])
        assert result.content == "Just some plain text response."


# ---------------------------------------------------------------------------
# _extract_json_object — balanced-brace parser
# ---------------------------------------------------------------------------

class TestExtractJsonObject:
    def test_simple_flat(self):
        from ovos_agentic_loop.react import _extract_json_object
        text = 'prefix {"key": "value"} suffix'
        assert _extract_json_object(text, 7) == '{"key": "value"}'

    def test_nested(self):
        from ovos_agentic_loop.react import _extract_json_object
        text = 'Action Input: {"a": {"b": 1}}'
        start = text.index("{")
        result = _extract_json_object(text, start)
        assert result == '{"a": {"b": 1}}'
        import json
        assert json.loads(result) == {"a": {"b": 1}}

    def test_brace_in_string_not_counted(self):
        from ovos_agentic_loop.react import _extract_json_object
        text = '{"key": "has } inside"}'
        result = _extract_json_object(text, 0)
        assert result == '{"key": "has } inside"}'

    def test_unterminated_returns_none(self):
        from ovos_agentic_loop.react import _extract_json_object
        assert _extract_json_object('{"unclosed": 1', 0) is None

    def test_parse_action_nested_json(self):
        from ovos_agentic_loop.react import _parse_action
        text = (
            'Action: search\n'
            'Action Input: {"filter": {"date": "today"}, "query": "test"}'
        )
        result = _parse_action(text)
        assert result is not None
        name, args = result
        assert name == "search"
        assert args["filter"] == {"date": "today"}


class TestBrainInjection:
    def test_set_brain_propagates_to_toolbox(self):
        from ovos_agentic_loop.react import ReActLoopEngine
        from unittest.mock import MagicMock
        engine = ReActLoopEngine(config={})
        tb = MagicMock()
        tb.set_brain = MagicMock()
        tb.tool_json_list = []
        engine.load_toolboxes([tb])
        brain = MagicMock()
        engine.set_brain(brain)
        tb.set_brain.assert_called_once_with(brain)

    def test_load_toolboxes_after_brain_also_injects(self):
        from ovos_agentic_loop.react import ReActLoopEngine
        from unittest.mock import MagicMock
        engine = ReActLoopEngine(config={})
        brain = MagicMock()
        engine._brain = brain  # set brain first
        tb = MagicMock()
        tb.set_brain = MagicMock()
        tb.tool_json_list = []
        engine.load_toolboxes([tb])
        tb.set_brain.assert_called_once_with(brain)

    def test_toolbox_without_set_brain_not_broken(self):
        from ovos_agentic_loop.react import ReActLoopEngine
        from unittest.mock import MagicMock
        engine = ReActLoopEngine(config={})
        tb = MagicMock(spec=[])  # no set_brain attribute
        tb.tool_json_list = []
        engine.load_toolboxes([tb])
        brain = MagicMock()
        engine.set_brain(brain)  # must not raise
