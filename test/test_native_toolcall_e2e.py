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

"""Real-stack end-to-end test for NativeToolCallEngine.

Wires the whole tool-calling stack with REAL classes and a single mocked boundary
(the LLM HTTP call):

    NativeToolCallEngine (ovos-agentic-loop)
      -> OpenAIChatEngine.continue_chat(tools=ToolBox)   (ovos-openai-plugin, real)
        -> ToolBox.normalize_tools -> /chat/completions   (HTTP mocked)
        <- structured tool_calls -> ToolCall              (real contract)
      -> MathToolBox.call_tool("evaluate_expression", ...) (real execution)
      -> MessageRole.TOOL result -> re-serialized          (real)
      -> /chat/completions                                  (HTTP mocked)
      <- final answer

Only ``requests.post`` inside the openai plugin is mocked; everything else —
the loop, the engine, message serialization, tool-call parsing, and the actual
math tool execution — is real.
"""
import json
from unittest.mock import MagicMock, patch

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_agentic_loop.native_toolcall import NativeToolCallEngine
from ovos_agentic_loop.tools.math import MathToolBox
from ovos_openai_plugin.chat import OpenAIChatEngine
import ovos_openai_plugin.api as openai_api


def _completion(message: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"choices": [{"message": message}]}
    return resp


def test_native_loop_executes_real_math_tool():
    """The model requests evaluate_expression; the real MathToolBox computes it."""
    # Turn 1: the LLM asks to call evaluate_expression("12*9").
    # Turn 2: with the tool result in context, it answers.
    responses = [
        _completion({
            "content": None,
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "evaluate_expression",
                             "arguments": json.dumps({"expression": "12*9"})},
            }],
        }),
        _completion({"content": "12 times 9 is 108."}),
    ]

    brain = OpenAIChatEngine({"api_url": "http://x/v1", "model": "test"})
    engine = NativeToolCallEngine()
    engine.set_brain(brain)
    engine.load_toolboxes([MathToolBox()])

    with patch.object(openai_api.requests, "post", side_effect=responses) as mock_post:
        result = engine.continue_chat(
            [AgentMessage(role=MessageRole.USER, content="what is 12*9?")])

    assert result.role == MessageRole.ASSISTANT
    assert "108" in result.content

    # Two real LLM round-trips happened.
    assert mock_post.call_count == 2

    # Turn 1 payload advertised the real toolbox schema as OpenAI tools.
    turn1 = json.loads(mock_post.call_args_list[0].kwargs["data"])
    tool_names = {t["function"]["name"] for t in turn1["tools"]}
    assert "evaluate_expression" in tool_names

    # Turn 2 payload carried the assistant tool_call turn + the real TOOL result
    # (the math tool actually computed 108).
    turn2 = json.loads(mock_post.call_args_list[1].kwargs["data"])
    roles = [m["role"] for m in turn2["messages"]]
    assert "assistant" in roles and "tool" in roles
    tool_msg = next(m for m in turn2["messages"] if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_1"
    assert "108" in tool_msg["content"]


def test_native_loop_falls_back_when_brain_has_no_tool_support():
    """A brain without supports_tools drives the inherited ReAct text loop."""
    from ovos_agentic_loop.react import _FINAL_ANSWER_TOKEN

    brain = OpenAIChatEngine({"api_url": "http://x/v1", "model": "test"})
    # OpenAIChatEngine *does* support tools; force the fallback path explicitly to
    # prove a non-tool brain still works end to end.
    brain.supports_tools = False
    engine = NativeToolCallEngine()
    engine.set_brain(brain)
    engine.load_toolboxes([MathToolBox()])

    responses = [_completion({"content": f"Thought: trivial.\n{_FINAL_ANSWER_TOKEN} 42"})]
    with patch.object(openai_api.requests, "post", side_effect=responses):
        result = engine.continue_chat(
            [AgentMessage(role=MessageRole.USER, content="6*7?")])
    assert result.content == "42"
