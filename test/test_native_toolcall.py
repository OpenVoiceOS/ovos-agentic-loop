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

"""Unit tests for NativeToolCallEngine."""
from typing import List
from unittest.mock import MagicMock

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole, ToolCall
from ovos_agentic_loop.native_toolcall import NativeToolCallEngine
from ovos_agentic_loop.react import _FINAL_ANSWER_TOKEN


def _tool_brain(responses: List[AgentMessage]) -> MagicMock:
    """Mock tool-capable ChatEngine returning the given AgentMessages in order."""
    brain = MagicMock()
    brain.supports_tools = True
    brain.continue_chat.side_effect = responses
    return brain


def _calc_toolbox(result: str = "3") -> MagicMock:
    """Mock ToolBox exposing a single 'calc' tool."""
    tb = MagicMock()
    tb.tool_json_list = [{
        "name": "calc",
        "description": "Add numbers.",
        "argument_schema": {"type": "object", "properties": {"a": {"type": "integer"}}},
    }]
    tb.get_tool.return_value = MagicMock()
    tb.call_tool.return_value = result
    return tb


class TestNativeToolCallEngine:
    def test_single_tool_then_answer(self) -> None:
        engine = NativeToolCallEngine()
        engine.set_brain(_tool_brain([
            AgentMessage(role=MessageRole.ASSISTANT, content="",
                         tool_calls=[ToolCall(id="c1", name="calc", arguments={"a": 1, "b": 2})]),
            AgentMessage(role=MessageRole.ASSISTANT, content="3"),
        ]))
        tb = _calc_toolbox("3")
        engine.load_toolboxes([tb])

        result = engine.continue_chat([AgentMessage(role=MessageRole.USER, content="1+2?")])
        assert result.role == MessageRole.ASSISTANT
        assert result.content == "3"
        tb.call_tool.assert_called_once_with("calc", {"a": 1, "b": 2})

        # The 2nd brain call must include the assistant tool_calls turn + a TOOL
        # result carrying the matching tool_call_id.
        second_call_messages = engine.brain.continue_chat.call_args_list[1].args[0]
        tool_msgs = [m for m in second_call_messages if m.role == MessageRole.TOOL]
        assert tool_msgs and tool_msgs[0].tool_call_id == "c1"
        assert tool_msgs[0].content == "3"
        assert any(m.role == MessageRole.ASSISTANT and m.tool_calls
                   for m in second_call_messages)

    def test_no_tool_calls_immediate_answer(self) -> None:
        engine = NativeToolCallEngine()
        engine.set_brain(_tool_brain([
            AgentMessage(role=MessageRole.ASSISTANT, content="hello"),
        ]))
        tb = _calc_toolbox()
        engine.load_toolboxes([tb])

        result = engine.continue_chat([AgentMessage(role=MessageRole.USER, content="hi")])
        assert result.content == "hello"
        tb.call_tool.assert_not_called()

    def test_multiple_tool_calls_one_turn(self) -> None:
        engine = NativeToolCallEngine()
        engine.set_brain(_tool_brain([
            AgentMessage(role=MessageRole.ASSISTANT, content="",
                         tool_calls=[ToolCall(id="c1", name="calc", arguments={"a": 1}),
                                     ToolCall(id="c2", name="calc", arguments={"a": 2})]),
            AgentMessage(role=MessageRole.ASSISTANT, content="done"),
        ]))
        tb = _calc_toolbox("ok")
        engine.load_toolboxes([tb])

        result = engine.continue_chat([AgentMessage(role=MessageRole.USER, content="go")])
        assert result.content == "done"
        assert tb.call_tool.call_count == 2
        second_call_messages = engine.brain.continue_chat.call_args_list[1].args[0]
        tool_ids = [m.tool_call_id for m in second_call_messages if m.role == MessageRole.TOOL]
        assert tool_ids == ["c1", "c2"]

    def test_max_iterations_forces_final_answer(self) -> None:
        engine = NativeToolCallEngine({"max_iterations": 2})
        # Always asks for a tool → loop exhausts, then one tool-free final call.
        tool_resp = lambda: AgentMessage(  # noqa: E731
            role=MessageRole.ASSISTANT, content="",
            tool_calls=[ToolCall(id="c", name="calc", arguments={})])
        engine.set_brain(_tool_brain([
            tool_resp(), tool_resp(),
            AgentMessage(role=MessageRole.ASSISTANT, content="forced answer"),
        ]))
        engine.load_toolboxes([_calc_toolbox()])

        result = engine.continue_chat([AgentMessage(role=MessageRole.USER, content="loop")])
        assert result.content == "forced answer"
        # 2 loop iterations + 1 final tool-free call
        assert engine.brain.continue_chat.call_count == 3
        assert engine.brain.continue_chat.call_args_list[-1].kwargs.get("tools") is None

    def test_fallback_to_react_when_brain_lacks_tools(self) -> None:
        engine = NativeToolCallEngine()
        brain = MagicMock()
        brain.supports_tools = False
        brain.continue_chat.side_effect = [
            AgentMessage(role=MessageRole.ASSISTANT,
                         content=f"Thought: easy.\n{_FINAL_ANSWER_TOKEN} 42"),
        ]
        engine.set_brain(brain)

        result = engine.continue_chat([AgentMessage(role=MessageRole.USER, content="6*7?")])
        # ReAct text loop parsed the FINAL_ANSWER token.
        assert result.content == "42"

    def test_no_brain_returns_error(self) -> None:
        engine = NativeToolCallEngine()
        result = engine.continue_chat([AgentMessage(role=MessageRole.USER, content="hi")])
        assert "Error" in result.content
