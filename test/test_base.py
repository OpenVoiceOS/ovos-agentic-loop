"""Unit tests for AgenticLoopEngine base class."""
from typing import List, Optional
from unittest.mock import MagicMock

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
