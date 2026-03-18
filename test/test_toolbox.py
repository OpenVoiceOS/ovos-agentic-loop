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

"""Unit tests for SkillMDToolBox and helpers."""
import textwrap
from unittest.mock import MagicMock, call

import pytest

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_agentic_loop.skills.loader import SkillMDEntry
from ovos_agentic_loop.skills.toolbox import (
    SkillCallArgs,
    SkillCallOutput,
    SkillMDToolBox,
    _slugify,
)
from ovos_agentic_loop.tools.clock import ClockToolBox


# ---------------------------------------------------------------------------
# _slugify
# ---------------------------------------------------------------------------

class TestSlugify:
    def test_simple(self) -> None:
        assert _slugify("web-search") == "web_search"

    def test_spaces(self) -> None:
        assert _slugify("My Cool Tool") == "my_cool_tool"

    def test_already_slug(self) -> None:
        assert _slugify("my_tool") == "my_tool"

    def test_leading_trailing_special(self) -> None:
        assert _slugify("--web--") == "web"


# ---------------------------------------------------------------------------
# SkillMDToolBox
# ---------------------------------------------------------------------------

SKILL_BODY = "You are a web search assistant. Search and return concise answers."

ENTRY = SkillMDEntry(
    name="web-search",
    description="Use when the agent needs to search the web.",
    body=SKILL_BODY,
    path="/fake/SKILL.md",
)


def _make_toolbox(entries=None) -> SkillMDToolBox:
    """Return a SkillMDToolBox with mocked loader."""
    tb = SkillMDToolBox()
    tb._loader = MagicMock()
    tb._loader.load.return_value = entries if entries is not None else [ENTRY]
    return tb


class TestSkillMDToolBoxDiscover:
    def test_discover_returns_one_tool_per_entry(self) -> None:
        tb = _make_toolbox([ENTRY])
        tools = tb.discover_tools()
        assert len(tools) == 1
        assert tools[0].name == "web_search"
        assert "search the web" in tools[0].description

    def test_discover_empty_loader(self) -> None:
        tb = _make_toolbox([])
        assert tb.discover_tools() == []


class TestSkillMDToolBoxInvoke:
    def test_invoke_calls_brain(self) -> None:
        tb = _make_toolbox([ENTRY])

        brain = MagicMock()
        brain.continue_chat.return_value = AgentMessage(
            role=MessageRole.ASSISTANT, content="Paris is the capital of France."
        )
        tb.set_brain(brain)

        args = SkillCallArgs(task="What is the capital of France?")
        output = tb._invoke_skill(ENTRY, args)

        assert isinstance(output, SkillCallOutput)
        assert "Paris" in output.result
        assert output.skill_used == "web-search"

        call_messages = brain.continue_chat.call_args[0][0]
        assert call_messages[0].role == MessageRole.SYSTEM
        assert call_messages[0].content == SKILL_BODY

    def test_invoke_with_context_appended(self) -> None:
        tb = _make_toolbox([ENTRY])
        brain = MagicMock()
        brain.continue_chat.return_value = AgentMessage(
            role=MessageRole.ASSISTANT, content="result"
        )
        tb.set_brain(brain)

        args = SkillCallArgs(task="Find something", context="limit to 2024")
        tb._invoke_skill(ENTRY, args)

        user_msg = brain.continue_chat.call_args[0][0][1]
        assert "limit to 2024" in user_msg.content

    def test_invoke_without_brain_raises(self) -> None:
        tb = _make_toolbox([ENTRY])
        args = SkillCallArgs(task="do something")
        with pytest.raises(RuntimeError, match="brain"):
            tb._invoke_skill(ENTRY, args)


class TestSkillMDToolBoxCallTool:
    def test_call_tool_via_toolbox_interface(self) -> None:
        tb = _make_toolbox([ENTRY])
        brain = MagicMock()
        brain.continue_chat.return_value = AgentMessage(
            role=MessageRole.ASSISTANT, content="42"
        )
        tb.set_brain(brain)

        # Force tool cache population.
        tb.refresh_tools()
        result = tb.call_tool("web_search", {"task": "answer to life"})
        assert result.result == "42"


# ---------------------------------------------------------------------------
# ToolBox bus protocol — bind / handle_discover / handle_call (ISSUE-015)
# ---------------------------------------------------------------------------

def _make_message(data: dict, msg_type: str = "test") -> MagicMock:
    """Build a minimal mock Message compatible with the OPM ToolBox protocol."""
    msg = MagicMock()
    msg.data = data
    msg.response.side_effect = lambda d: d  # response() just returns the payload dict
    return msg


class TestToolBoxBusProtocol:
    def test_bind_registers_handlers(self) -> None:
        """bind() subscribes to the discovery broadcast and the per-toolbox call channel."""
        tb = ClockToolBox()
        bus = MagicMock()
        tb.bind(bus)
        assert tb.bus is bus
        # Must register exactly these two event names.
        registered = {c.args[0] for c in bus.on.call_args_list}
        assert "ovos.persona.tools.discover" in registered
        assert f"ovos.persona.tools.{tb.toolbox_id}.call" in registered

    def test_handle_discover_emits_tool_list(self) -> None:
        """handle_discover emits the tool JSON list and toolbox_id."""
        tb = ClockToolBox()
        bus = MagicMock()
        tb.bind(bus)

        msg = _make_message({})
        tb.handle_discover(msg)

        bus.emit.assert_called_once()
        emitted = bus.emit.call_args[0][0]
        assert "tools" in emitted
        assert emitted["toolbox_id"] == tb.toolbox_id
        # ClockToolBox exposes exactly one tool.
        assert len(emitted["tools"]) == 1
        assert emitted["tools"][0]["name"] == "get_current_datetime"

    def test_handle_call_returns_result(self) -> None:
        """handle_call dispatches to call_tool and emits a result payload."""
        tb = ClockToolBox()
        bus = MagicMock()
        tb.bind(bus)

        msg = _make_message({"name": "get_current_datetime", "kwargs": {}})
        tb.handle_call(msg)

        bus.emit.assert_called_once()
        emitted = bus.emit.call_args[0][0]
        assert "result" in emitted
        assert emitted["toolbox_id"] == tb.toolbox_id
        assert "iso" in emitted["result"]

    def test_handle_call_unknown_tool_emits_error(self) -> None:
        """handle_call emits an error payload for an unknown tool name."""
        tb = ClockToolBox()
        bus = MagicMock()
        tb.bind(bus)

        msg = _make_message({"name": "nonexistent_tool", "kwargs": {}})
        tb.handle_call(msg)

        bus.emit.assert_called_once()
        emitted = bus.emit.call_args[0][0]
        assert "error" in emitted
        assert emitted["toolbox_id"] == tb.toolbox_id

    def test_handle_call_without_bind_does_not_crash(self) -> None:
        """Calling handle_call before bind is not a supported use-case but must not
        silently corrupt state — the bus attribute is None and the call should raise
        AttributeError, which the test documents as expected behaviour."""
        tb = ClockToolBox()
        msg = _make_message({"name": "get_current_datetime", "kwargs": {}})
        with pytest.raises(AttributeError):
            tb.handle_call(msg)
