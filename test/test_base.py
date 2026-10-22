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

"""Unit tests for AgenticLoopEngine base class."""
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_agentic_loop.base import AgenticLoopEngine


class _ConcreteLoopEngine(AgenticLoopEngine):
    """Minimal concrete subclass for testing the abstract base."""

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        return AgentMessage(role=MessageRole.ASSISTANT, content="ok")


class TestAgenticLoopEngineInit:
    def test_default_toolboxes_empty(self) -> None:
        engine = _ConcreteLoopEngine()
        assert engine.toolboxes == []

    def test_config_stored(self) -> None:
        engine = _ConcreteLoopEngine(config={"key": "val"})
        assert engine.config["key"] == "val"


class TestLoadToolboxes:
    def test_load_toolboxes_replaces_list(self) -> None:
        engine = _ConcreteLoopEngine()
        tb1, tb2 = MagicMock(), MagicMock()
        engine.load_toolboxes([tb1, tb2])
        assert engine.toolboxes == [tb1, tb2]

    def test_load_toolboxes_empty_clears(self) -> None:
        engine = _ConcreteLoopEngine()
        engine.load_toolboxes([MagicMock()])
        engine.load_toolboxes([])
        assert engine.toolboxes == []


class TestContinueChat:
    def test_returns_assistant_message(self) -> None:
        engine = _ConcreteLoopEngine()
        msg = AgentMessage(role=MessageRole.USER, content="hello")
        result = engine.continue_chat([msg])
        assert result.role == MessageRole.ASSISTANT


class TestAbstractEnforcement:
    def test_cannot_instantiate_base_directly(self) -> None:
        with pytest.raises(TypeError):
            AgenticLoopEngine()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# _load_toolboxes_from_config — OPM plugin path (ISSUE-013)
# ---------------------------------------------------------------------------

class TestLoadToolboxesFromConfigOPM:
    def test_loads_toolbox_via_opm(self) -> None:
        mock_toolbox = MagicMock()
        mock_toolbox_cls = MagicMock(return_value=mock_toolbox)

        with patch.dict("sys.modules", {
            "ovos_plugin_manager.persona": MagicMock(
                find_toolbox_plugins=MagicMock(return_value={"my-toolbox": mock_toolbox_cls})
            )
        }):
            engine = _ConcreteLoopEngine(config={"toolboxes": ["my-toolbox"]})

        assert mock_toolbox in engine.toolboxes
        mock_toolbox_cls.assert_called_once_with(config={}, bus=None)

    def test_skips_gracefully_when_opm_unavailable(self) -> None:
        with patch.dict("sys.modules", {"ovos_plugin_manager.persona": None}):
            engine = _ConcreteLoopEngine(config={"toolboxes": ["my-toolbox"]})
        assert engine.toolboxes == []

    def test_warns_on_toolbox_load_failure(self, capsys: "pytest.CaptureFixture") -> None:
        with patch.dict("sys.modules", {
            "ovos_plugin_manager.persona": MagicMock(
                find_toolbox_plugins=MagicMock(return_value={})
            )
        }):
            engine = _ConcreteLoopEngine(config={"toolboxes": ["bad-tb"]})

        assert engine.toolboxes == []
        # OVOS LOG writes to stdout; confirm the warning was emitted.
        captured = capsys.readouterr()
        assert "bad-tb" in captured.out
