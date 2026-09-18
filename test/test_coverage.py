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

"""Supplemental tests targeting uncovered branches for 95%+ coverage."""
import os
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

from ovos_agentic_loop.context.agents_md import AgentsMDContextManager, _discover_agents_md_paths
from ovos_agentic_loop.skills.loader import _discover_via_entry_points, _discover_via_package_data
from ovos_agentic_loop.tools.filesystem import FileSystemToolBox
from ovos_agentic_loop.tools.web import WebSearchToolBox
from ovos_agentic_loop.react import _extract_json_object, _parse_action
from ovos_agentic_loop.chain_of_thought import ChainOfThoughtEngine
from ovos_agentic_loop.critic import CRITICEngine
from ovos_agentic_loop.self_ask import SelfAskEngine, _extract_follow_up, _extract_final_answer
from ovos_agentic_loop.plan_execute import PlanAndExecuteEngine
from ovos_agentic_loop.reflexion import ReflexionEngine
from ovos_agentic_loop.base import AgenticLoopEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role: MessageRole, content: str) -> AgentMessage:
    return AgentMessage(role=role, content=content)


def _brain(responses: List[str]) -> MagicMock:
    b = MagicMock()
    b.continue_chat.side_effect = [
        _msg(MessageRole.ASSISTANT, r) for r in responses
    ]
    return b


# ---------------------------------------------------------------------------
# base.py — brain injection exception path and OPM load path
# ---------------------------------------------------------------------------

class TestAgenticLoopEngineBaseCoverage:
    """Cover the error/OPM paths in base.py."""

    def test_inject_brain_swallows_set_brain_exception(self) -> None:
        """_inject_brain_into_toolboxes logs warning when set_brain raises."""
        engine = ChainOfThoughtEngine()
        bad_tb = MagicMock()
        bad_tb.set_brain.side_effect = RuntimeError("boom")
        engine.toolboxes = [bad_tb]
        # Should not raise
        engine._inject_brain_into_toolboxes(MagicMock())

    def test_load_toolboxes_from_config_no_ids(self) -> None:
        """_load_toolboxes_from_config does nothing when config has no toolboxes key."""
        engine = ChainOfThoughtEngine(config={})
        engine._load_toolboxes_from_config()  # should not raise

    def test_load_toolboxes_from_config_opm_import_error(self) -> None:
        """_load_toolboxes_from_config handles missing OPM gracefully."""
        engine = ChainOfThoughtEngine(config={"toolboxes": ["some-tool"]})
        with patch.dict("sys.modules", {"ovos_plugin_manager.persona": None}):
            engine._load_toolboxes_from_config()  # should not raise

    def test_load_toolboxes_from_config_plugin_load_fails(self) -> None:
        """_load_toolboxes_from_config logs warning on per-plugin failure."""
        engine = ChainOfThoughtEngine(config={"toolboxes": ["bad-tool"]})
        mock_opm = MagicMock()
        mock_opm.find_toolbox_plugins.return_value = {}
        with patch.dict("sys.modules", {"ovos_plugin_manager.persona": mock_opm}):
            engine._load_toolboxes_from_config()  # should not raise

    def test_load_toolboxes_with_brain_already_set(self) -> None:
        """load_toolboxes() injects brain into newly loaded toolboxes."""
        engine = ChainOfThoughtEngine()
        engine._brain = MagicMock()
        tb = MagicMock()
        engine.load_toolboxes([tb])
        tb.set_brain.assert_called_once_with(engine._brain)


# ---------------------------------------------------------------------------
# context/agents_md.py — _discover_agents_md_paths branches
# ---------------------------------------------------------------------------

class TestDiscoverAgentsMdPaths:
    def test_auto_discovery_returns_list(self) -> None:
        """_discover_agents_md_paths returns a list (may be empty)."""
        result = _discover_agents_md_paths()
        assert isinstance(result, list)

    def test_distributions_exception_returns_empty(self) -> None:
        """_discover_agents_md_paths returns [] when distributions() raises."""
        with patch("importlib.metadata.distributions", side_effect=RuntimeError("fail")):
            result = _discover_agents_md_paths()
        assert result == []

    def test_dist_files_exception_is_skipped(self) -> None:
        """_discover_agents_md_paths skips distributions whose files raise."""
        bad_dist = MagicMock()
        bad_dist.files = property(lambda self: (_ for _ in ()).throw(RuntimeError("bad")))
        with patch("importlib.metadata.distributions", return_value=[bad_dist]):
            result = _discover_agents_md_paths()
        assert isinstance(result, list)

    def test_auto_source_triggers_discovery(self, tmp_path: Path) -> None:
        """agents_md_sources=['auto'] loads from package data discovery."""
        p = tmp_path / "AGENTS.md"
        p.write_text("# S\nbody", encoding="utf-8")
        with patch(
            "ovos_agentic_loop.context.agents_md._discover_agents_md_paths",
            return_value=[str(p)],
        ):
            mgr = AgentsMDContextManager(config={"agents_md_sources": ["auto"]})
            assert "S" in mgr.system_prompt

    def test_oserror_during_file_read_is_skipped(self, tmp_path: Path) -> None:
        """OSError when reading an AGENTS.md file is silently skipped."""
        p = tmp_path / "AGENTS.md"
        p.write_text("# S\nbody", encoding="utf-8")
        mgr = AgentsMDContextManager(config={"agents_md_sources": [str(p)]})
        with patch("ovos_agentic_loop.context.agents_md.open", side_effect=OSError("disk full")):
            mgr.invalidate_cache()
            prompt = mgr.system_prompt
        assert prompt == ""


# ---------------------------------------------------------------------------
# skills/loader.py — entry-point and package-data discovery branches
# ---------------------------------------------------------------------------

class TestSkillMDLoaderDiscovery:
    def test_discover_via_entry_points_returns_list(self) -> None:
        result = _discover_via_entry_points()
        assert isinstance(result, list)

    def test_discover_via_entry_points_ep_exception(self) -> None:
        """entry_points() failure returns empty list."""
        with patch("importlib.metadata.entry_points", side_effect=RuntimeError):
            result = _discover_via_entry_points()
        assert result == []

    def test_discover_via_entry_points_ep_load_fallback(self, tmp_path: Path) -> None:
        """ep.load() failure falls back to ep.value as direct path."""
        p = tmp_path / "SKILL.md"
        p.write_text("# skill", encoding="utf-8")
        ep = MagicMock()
        ep.load.side_effect = ImportError("bad")
        ep.value = str(p)
        with patch("importlib.metadata.entry_points", return_value=[ep]):
            result = _discover_via_entry_points()
        assert str(p) in result

    def test_discover_via_package_data_returns_list(self) -> None:
        result = _discover_via_package_data()
        assert isinstance(result, list)

    def test_discover_via_package_data_distributions_exception(self) -> None:
        with patch("importlib.metadata.distributions", side_effect=RuntimeError):
            result = _discover_via_package_data()
        assert result == []

    def test_discover_via_package_data_dist_files_exception(self) -> None:
        bad_dist = MagicMock()
        type(bad_dist).files = property(lambda self: (_ for _ in ()).throw(RuntimeError("bad")))
        with patch("importlib.metadata.distributions", return_value=[bad_dist]):
            result = _discover_via_package_data()
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# tools/filesystem.py — more branch coverage
# ---------------------------------------------------------------------------

class TestFileSystemToolBoxAdditional:
    def _box(self, tmp_path: Path, **cfg) -> FileSystemToolBox:
        return FileSystemToolBox(config={"root_path": str(tmp_path), **cfg})

    def test_write_traversal_blocked(self, tmp_path: Path) -> None:
        result = self._box(tmp_path).call_tool("write_file", {"path": "../../evil.txt", "content": "x"})
        assert result.success is False
        assert "Access denied" in result.message

    def test_list_directory_traversal_blocked(self, tmp_path: Path) -> None:
        result = self._box(tmp_path).call_tool("list_directory", {"path": "../../"})
        assert result.entries == []

    def test_search_traversal_blocked(self, tmp_path: Path) -> None:
        result = self._box(tmp_path).call_tool("search_in_files", {"pattern": "x", "path": "../../"})
        assert result.total == 0

    def test_find_traversal_blocked(self, tmp_path: Path) -> None:
        result = self._box(tmp_path).call_tool("find_files", {"glob": "*.py", "path": "../../"})
        assert result.total == 0

    def test_write_creates_parent_directories(self, tmp_path: Path) -> None:
        result = self._box(tmp_path).call_tool("write_file", {"path": "a/b/c.txt", "content": "hi"})
        assert result.success is True
        assert (tmp_path / "a" / "b" / "c.txt").read_text() == "hi"


# ---------------------------------------------------------------------------
# react.py — escape sequences in JSON (lines 76-80)
# ---------------------------------------------------------------------------

class TestFileSystemToolBoxErrorPaths:
    def _box(self, tmp_path: Path, **cfg) -> FileSystemToolBox:
        return FileSystemToolBox(config={"root_path": str(tmp_path), **cfg})

    def test_write_exception_returns_failure(self, tmp_path: Path) -> None:
        """write_file returns success=False when write_text raises."""
        box = self._box(tmp_path)
        with patch("pathlib.Path.write_text", side_effect=OSError("no space")):
            result = box.call_tool("write_file", {"path": "out.txt", "content": "x"})
        assert result.success is False
        assert "Error" in result.message

    def test_list_directory_glob_exception(self, tmp_path: Path) -> None:
        """list_directory returns empty list when glob raises."""
        box = self._box(tmp_path)
        with patch("pathlib.Path.glob", side_effect=OSError("bad")):
            result = box.call_tool("list_directory", {"path": "."})
        assert result.entries == []

    def test_search_in_files_invalid_regex(self, tmp_path: Path) -> None:
        """search_in_files returns empty results for invalid regex."""
        (tmp_path / "f.txt").write_text("hello")
        result = self._box(tmp_path).call_tool("search_in_files", {"pattern": "[invalid", "path": "."})
        assert result.total == 0

    def test_read_file_exception_returns_error_msg(self, tmp_path: Path) -> None:
        """read_file returns an error message when read raises unexpectedly."""
        (tmp_path / "f.txt").write_text("x")
        box = self._box(tmp_path)
        with patch("pathlib.Path.read_text", side_effect=PermissionError("denied")):
            result = box.call_tool("read_file", {"path": "f.txt"})
        assert "Error" in result.content


class TestExtractJsonObjectEscapes:
    def test_backslash_escape_in_string(self) -> None:
        """Backslash-escaped quote inside a string must not close the object."""
        text = '{"key": "value with \\"quote\\" inside"}'
        result = _extract_json_object(text, 0)
        assert result == text

    def test_parse_action_unterminated_json_returns_none(self) -> None:
        """_parse_action returns None when the JSON object is not terminated."""
        text = "Action: search\nAction Input: {\"query\": \"missing close brace\""
        assert _parse_action(text) is None

    def test_nested_object_extracted_completely(self) -> None:
        text = 'prefix {"outer": {"inner": 1}} suffix'
        start = text.index("{")
        result = _extract_json_object(text, start)
        assert result == '{"outer": {"inner": 1}}'


# ---------------------------------------------------------------------------
# chain_of_thought.py — _load_brain with brain config set
# ---------------------------------------------------------------------------

class TestChainOfThoughtLoadBrain:
    def test_load_brain_returns_none_when_no_brain_config(self) -> None:
        engine = ChainOfThoughtEngine(config={})
        assert engine._load_brain() is None

    def test_load_brain_returns_none_on_import_error(self) -> None:
        engine = ChainOfThoughtEngine(config={"brain": "some-brain-id"})
        with patch.dict("sys.modules", {"ovos_plugin_manager.agents": None}):
            result = engine._load_brain()
        assert result is None


# ---------------------------------------------------------------------------
# critic.py — _call_tool not found and _load_brain
# ---------------------------------------------------------------------------

class TestCRITICCoverage:
    def test_call_tool_not_found(self) -> None:
        engine = CRITICEngine()
        engine._brain = _brain(["Final answer"])
        result = engine._call_tool("nonexistent", {"q": "x"})
        assert "not found" in result

    def test_call_tool_found(self) -> None:
        engine = CRITICEngine()
        tb = MagicMock()
        tb.get_tool.return_value = MagicMock()
        tb.call_tool.return_value = MagicMock(__str__=lambda self: "tool result")
        engine.toolboxes = [tb]
        result = engine._call_tool("my_tool", {"q": "x"})
        assert result == "tool result"

    def test_load_brain_returns_none_on_exception(self) -> None:
        engine = CRITICEngine(config={"brain": "bad-id"})
        with patch.dict("sys.modules", {"ovos_plugin_manager.agents": None}):
            assert engine._load_brain() is None

    def test_call_tool_exception_continues(self) -> None:
        """_call_tool exception in get_tool is swallowed."""
        engine = CRITICEngine()
        tb = MagicMock()
        tb.get_tool.side_effect = RuntimeError("boom")
        engine.toolboxes = [tb]
        result = engine._call_tool("t", {"q": "x"})
        assert "not found" in result

    def test_critic_early_exit_on_verified(self) -> None:
        """CRITIC loop exits early when brain returns ALL_VERIFIED sentinel."""
        engine = CRITICEngine(config={"max_critique_rounds": 3})
        engine._brain = _brain([
            "Draft answer.",
            "ALL_VERIFIED",
        ])
        msgs = [_msg(MessageRole.USER, "what is 2+2?")]
        result = engine.continue_chat(msgs)
        assert result.role == MessageRole.ASSISTANT


# ---------------------------------------------------------------------------
# self_ask.py — _call_first_matching_tool and _call_named_tool paths
# ---------------------------------------------------------------------------

class TestSelfAskCoverage:
    def test_call_first_matching_tool_no_toolboxes(self) -> None:
        engine = SelfAskEngine()
        result = engine._call_first_matching_tool("what?")
        assert "No tool" in result

    def test_call_first_matching_tool_success(self) -> None:
        engine = SelfAskEngine()
        tb = MagicMock()
        mock_tool = MagicMock()
        mock_tool.name = "search"
        tb.discover_tools.return_value = [mock_tool]
        tb.call_tool.return_value = MagicMock(__str__=lambda self: "found it")
        engine.toolboxes = [tb]
        result = engine._call_first_matching_tool("query")
        assert result == "found it"

    def test_call_named_tool_not_found(self) -> None:
        engine = SelfAskEngine()
        result = engine._call_named_tool("missing_tool", "query")
        assert "not found" in result

    def test_call_named_tool_found(self) -> None:
        engine = SelfAskEngine()
        tb = MagicMock()
        tb.get_tool.return_value = MagicMock()
        tb.call_tool.return_value = MagicMock(__str__=lambda self: "answer")
        engine.toolboxes = [tb]
        result = engine._call_named_tool("my_tool", "query")
        assert result == "answer"

    def test_load_brain_returns_none_on_exception(self) -> None:
        engine = SelfAskEngine(config={"brain": "bad-id"})
        with patch.dict("sys.modules", {"ovos_plugin_manager.agents": None}):
            assert engine._load_brain() is None

    def test_call_first_matching_tool_empty_tools_list(self) -> None:
        """Toolbox with no tools is skipped."""
        engine = SelfAskEngine()
        tb = MagicMock()
        tb.discover_tools.return_value = []
        engine.toolboxes = [tb]
        result = engine._call_first_matching_tool("query")
        assert "No tool" in result

    def test_call_first_matching_tool_all_keys_fail(self) -> None:
        """All key attempts fail → falls through to next toolbox (no result)."""
        engine = SelfAskEngine()
        tb = MagicMock()
        mock_tool = MagicMock()
        mock_tool.name = "search"
        tb.discover_tools.return_value = [mock_tool]
        tb.call_tool.side_effect = Exception("nope")
        engine.toolboxes = [tb]
        result = engine._call_first_matching_tool("query")
        assert "No tool" in result

    def test_call_named_tool_all_keys_fail(self) -> None:
        """All key attempts fail for a named tool."""
        engine = SelfAskEngine()
        tb = MagicMock()
        tb.get_tool.return_value = MagicMock()
        tb.call_tool.side_effect = Exception("nope")
        engine.toolboxes = [tb]
        result = engine._call_named_tool("my_tool", "query")
        assert "not found" in result


# ---------------------------------------------------------------------------
# plan_execute.py — _call_tool and _load_brain
# ---------------------------------------------------------------------------

class TestPlanExecuteCoverage:
    def test_call_tool_not_found(self) -> None:
        engine = PlanAndExecuteEngine()
        engine._brain = _brain(["done"])
        result = engine._call_tool("missing", {})
        assert "not found" in result

    def test_call_tool_found(self) -> None:
        engine = PlanAndExecuteEngine()
        tb = MagicMock()
        tb.get_tool.return_value = MagicMock()
        tb.call_tool.return_value = MagicMock(__str__=lambda self: "ok")
        engine.toolboxes = [tb]
        result = engine._call_tool("tool", {"q": "x"})
        assert result == "ok"

    def test_load_brain_returns_none_on_exception(self) -> None:
        engine = PlanAndExecuteEngine(config={"brain": "bad-id"})
        with patch.dict("sys.modules", {"ovos_plugin_manager.agents": None}):
            assert engine._load_brain() is None

    def test_empty_plan_returns_no_steps(self) -> None:
        """An empty plan string falls through without error."""
        from ovos_agentic_loop.plan_execute import _parse_plan
        assert _parse_plan("No steps here.") == []

    def test_execute_step_with_tool_call(self) -> None:
        """_execute_step handles a tool action and observation loop."""
        from ovos_agentic_loop.plan_execute import PlanAndExecuteEngine
        engine = PlanAndExecuteEngine(config={"max_step_iterations": 2})
        tool_response = 'Action: search\nAction Input: {"query": "hello"}'
        engine._brain = _brain([
            tool_response,
            "Result: found it",
        ])
        tb = MagicMock()
        tb.get_tool.return_value = MagicMock()
        tb.call_tool.return_value = MagicMock(__str__=lambda self: "ok")
        engine.toolboxes = [tb]
        result = engine._execute_step("step 1", "plan", "", 1, [], "s", None, None)
        assert result is not None


# ---------------------------------------------------------------------------
# reflexion.py — _load_brain
# ---------------------------------------------------------------------------

class TestReflexionCoverage:
    def test_load_brain_returns_none_on_exception(self) -> None:
        engine = ReflexionEngine(config={"brain": "bad-id"})
        with patch.dict("sys.modules", {"ovos_plugin_manager.agents": None}):
            assert engine._load_brain() is None

    def test_set_brain_wires_inner_react(self) -> None:
        engine = ReflexionEngine()
        brain = _brain(["Final Answer: done."] * 10)
        # Lazily create the inner react engine first, then set_brain propagates.
        engine._get_react_engine()
        engine.set_brain(brain)
        assert engine._inner_react is not None
        assert engine._inner_react.brain is brain
