"""Tool-dispatch error handling for the loop engines.

When a tool is *found* but its execution raises, ``_call_tool`` must surface the real
error as the observation (so the LLM can recover within the loop) instead of swallowing
it and reporting the misleading "tool not found". A genuine lookup miss still falls
through to the next toolbox and finally reports "not found".

Covers ReActLoopEngine (also the base for NativeToolCallEngine, which inherits
``_call_tool``) and PlanAndExecuteEngine.
"""
from unittest.mock import MagicMock

import pytest

from ovos_plugin_manager.templates.agent_tools import AgentTool, ToolArguments, ToolBox, ToolOutput

from ovos_agentic_loop.react import ReActLoopEngine
from ovos_agentic_loop.native_toolcall import NativeToolCallEngine
from ovos_agentic_loop.plan_execute import PlanAndExecuteEngine


def _toolbox(has_tool, result=None, error=None):
    tb = MagicMock()
    tb.get_tool.return_value = MagicMock() if has_tool else None
    if error is not None:
        tb.call_tool.side_effect = error
    else:
        tb.call_tool.return_value = result
    return tb


@pytest.fixture(params=[ReActLoopEngine, NativeToolCallEngine, PlanAndExecuteEngine])
def engine(request):
    eng = request.param()
    eng.toolboxes = []
    return eng


def test_tool_found_and_succeeds(engine):
    engine.toolboxes = [_toolbox(has_tool=True, result="42")]
    assert engine._call_tool("calc", {"a": 1}) == "42"


def test_tool_found_but_raises_surfaces_error(engine):
    engine.toolboxes = [_toolbox(has_tool=True, error=RuntimeError("boom"))]
    obs = engine._call_tool("calc", {"a": 1})
    # the real failure is reported, NOT "not found"
    assert "failed" in obs and "boom" in obs
    assert "not found" not in obs


def test_tool_not_found_reports_not_found(engine):
    engine.toolboxes = [_toolbox(has_tool=False)]
    obs = engine._call_tool("missing", {})
    assert "not found" in obs


def test_falls_through_to_toolbox_that_has_tool(engine):
    first = _toolbox(has_tool=False)
    second = _toolbox(has_tool=True, result="ok")
    engine.toolboxes = [first, second]
    assert engine._call_tool("calc", {}) == "ok"
    second.call_tool.assert_called_once()


def test_lookup_error_falls_through_to_next_toolbox(engine):
    broken = MagicMock()
    broken.get_tool.side_effect = RuntimeError("toolbox down")
    healthy = _toolbox(has_tool=True, result="recovered")
    engine.toolboxes = [broken, healthy]
    assert engine._call_tool("calc", {}) == "recovered"


def test_execution_error_does_not_fall_through(engine):
    """A found-but-failing tool reports its error; later toolboxes are not consulted."""
    failing = _toolbox(has_tool=True, error=ValueError("bad args"))
    fallback = _toolbox(has_tool=True, result="should-not-be-used")
    engine.toolboxes = [failing, fallback]
    obs = engine._call_tool("calc", {})
    assert "bad args" in obs
    fallback.call_tool.assert_not_called()


class _Args(ToolArguments):
    text: str


class _Out(ToolOutput):
    text: str


class _FlakyToolBox(ToolBox):
    """A real OPM ToolBox whose one tool fails the way a dead server does."""

    def __init__(self):
        super().__init__(toolbox_id="flaky")

    def discover_tools(self):
        def _fetch(args):
            raise ConnectionError("connection refused by 127.0.0.1:9686")
        return [AgentTool(name="fetch", description="Fetch text.",
                          argument_schema=_Args, output_schema=_Out, tool_call=_fetch)]


def test_observation_names_the_root_cause_behind_toolbox_wrapping(engine):
    engine.toolboxes = [_FlakyToolBox()]
    obs = engine._call_tool("fetch", {"text": "hi"})
    assert obs.startswith("Error: tool 'fetch' failed: ")
    assert "ConnectionError: connection refused by 127.0.0.1:9686" in obs


def test_observation_names_the_invalid_argument(engine):
    engine.toolboxes = [_FlakyToolBox()]
    obs = engine._call_tool("fetch", {"txt": "hi"})
    assert "Tool input validation failed for 'fetch'" in obs
    assert "text" in obs.split("ValidationError", 1)[1]
    assert "Field required" in obs


@pytest.mark.timeout(10)
def test_cyclic_cause_chain_does_not_hang(engine):
    first, second = ValueError("a"), KeyError("b")
    first.__cause__, second.__cause__ = second, first
    engine.toolboxes = [_toolbox(has_tool=True, error=first)]
    obs = engine._call_tool("calc", {})
    assert obs == "Error: tool 'calc' failed: a: KeyError: 'b'"
