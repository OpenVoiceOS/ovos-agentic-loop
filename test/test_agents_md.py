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

"""Unit tests for AgentsMDContextManager."""
import textwrap
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole
from ovos_agentic_loop.context.agents_md import (
    AgentsMDContextManager,
    _discover_agents_md_paths,
    _parse_sections,
)


# ---------------------------------------------------------------------------
# _parse_sections
# ---------------------------------------------------------------------------

SAMPLE_MD = textwrap.dedent("""\
    # Project Overview

    This is the overview.

    ## Universal Rules

    - Rule 1
    - Rule 2

    ## OpenVoiceOS Workspace

    OVOS-specific content.
""")


class TestParseSections:
    def test_parses_all_headings(self) -> None:
        sections = _parse_sections(SAMPLE_MD)
        assert "Project Overview" in sections
        assert "Universal Rules" in sections
        assert "OpenVoiceOS Workspace" in sections

    def test_body_content(self) -> None:
        sections = _parse_sections(SAMPLE_MD)
        assert "Rule 1" in sections["Universal Rules"]
        assert "OVOS-specific" in sections["OpenVoiceOS Workspace"]

    def test_empty_document(self) -> None:
        assert _parse_sections("") == {}


# ---------------------------------------------------------------------------
# AgentsMDContextManager
# ---------------------------------------------------------------------------

class TestAgentsMDContextManagerSystemPrompt:
    def test_loads_from_extra_path(self, tmp_path: "Path") -> None:
        p = tmp_path / "AGENTS.md"
        p.write_text(SAMPLE_MD, encoding="utf-8")
        mgr = AgentsMDContextManager(
            config={"agents_md_sources": [str(p)]},
        )
        prompt = mgr.system_prompt
        assert "Universal Rules" in prompt
        assert "OpenVoiceOS Workspace" in prompt

    def test_include_sections_filter(self, tmp_path: "Path") -> None:
        p = tmp_path / "AGENTS.md"
        p.write_text(SAMPLE_MD, encoding="utf-8")
        mgr = AgentsMDContextManager(
            config={
                "agents_md_sources": [str(p)],
                "include_sections": ["Universal Rules"],
            },
        )
        prompt = mgr.system_prompt
        assert "Universal Rules" in prompt
        assert "OpenVoiceOS Workspace" not in prompt

    def test_prefix_prepended(self, tmp_path: "Path") -> None:
        p = tmp_path / "AGENTS.md"
        p.write_text(SAMPLE_MD, encoding="utf-8")
        mgr = AgentsMDContextManager(
            config={
                "agents_md_sources": [str(p)],
                "system_prompt_prefix": "You are a helpful assistant.",
            },
        )
        assert mgr.system_prompt.startswith("You are a helpful assistant.")

    def test_missing_file_returns_empty_body(self, tmp_path: "Path") -> None:
        mgr = AgentsMDContextManager(
            config={"agents_md_sources": [str(tmp_path / "nonexistent.md")]},
        )
        assert mgr.system_prompt == ""

    def test_cache_invalidation(self, tmp_path: "Path") -> None:
        p = tmp_path / "AGENTS.md"
        p.write_text(SAMPLE_MD, encoding="utf-8")
        mgr = AgentsMDContextManager(config={"agents_md_sources": [str(p)]})
        first = mgr.system_prompt
        mgr.invalidate_cache()
        p.write_text("# New Section\n\nnew content", encoding="utf-8")
        second = mgr.system_prompt
        assert first != second


class TestAgentsMDContextManagerHistory:
    def test_get_history_empty(self) -> None:
        mgr = AgentsMDContextManager()
        assert mgr.get_history("sess-1") == []

    def test_sessions_are_isolated(self) -> None:
        mgr = AgentsMDContextManager()
        msg = AgentMessage(role=MessageRole.USER, content="hello")
        mgr.update_history([msg], "sess-A")
        assert mgr.get_history("sess-A") == [msg]
        assert mgr.get_history("sess-B") == []

    def test_update_history(self) -> None:
        mgr = AgentsMDContextManager()
        msg = AgentMessage(role=MessageRole.USER, content="hello")
        mgr.update_history([msg], "s1")
        assert len(mgr.get_history("s1")) == 1
        assert mgr.get_history("s1")[0].content == "hello"

    def test_update_history_appends(self) -> None:
        mgr = AgentsMDContextManager()
        m1 = AgentMessage(role=MessageRole.USER, content="a")
        m2 = AgentMessage(role=MessageRole.ASSISTANT, content="b")
        mgr.update_history([m1], "s")
        mgr.update_history([m2], "s")
        assert len(mgr.get_history("s")) == 2

    def test_get_history_returns_copy(self) -> None:
        mgr = AgentsMDContextManager()
        msg = AgentMessage(role=MessageRole.USER, content="hi")
        mgr.update_history([msg], "s")
        history = mgr.get_history("s")
        history.clear()
        assert len(mgr.get_history("s")) == 1


class TestAgentsMDContextManagerBuildContext:
    def test_builds_correct_order(self, tmp_path: "Path") -> None:
        p = tmp_path / "AGENTS.md"
        p.write_text(SAMPLE_MD, encoding="utf-8")
        mgr = AgentsMDContextManager(config={"agents_md_sources": [str(p)]})

        user_msg = AgentMessage(role=MessageRole.USER, content="previous")
        mgr.update_history([user_msg], "sess")

        messages = mgr.build_conversation_context("new question", "sess")
        roles = [m.role for m in messages]
        assert roles[0] == MessageRole.SYSTEM
        assert roles[-1] == MessageRole.USER
        assert messages[-1].content == "new question"

    def test_history_included_in_context(self, tmp_path: "Path") -> None:
        p = tmp_path / "AGENTS.md"
        p.write_text("# S\nbody", encoding="utf-8")
        mgr = AgentsMDContextManager(config={"agents_md_sources": [str(p)]})
        mgr.update_history([AgentMessage(role=MessageRole.USER, content="prev")], "s")
        msgs = mgr.build_conversation_context("now", "s")
        contents = [m.content for m in msgs]
        assert "prev" in contents
        assert "now" in contents

    def test_different_sessions_get_different_history(self) -> None:
        mgr = AgentsMDContextManager(config={"agents_md_sources": []})
        mgr.update_history([AgentMessage(role=MessageRole.USER, content="A-msg")], "A")
        msgs_a = mgr.build_conversation_context("q", "A")
        msgs_b = mgr.build_conversation_context("q", "B")
        assert any(m.content == "A-msg" for m in msgs_a)
        assert not any(m.content == "A-msg" for m in msgs_b)

    def test_no_system_prompt_when_empty(self) -> None:
        mgr = AgentsMDContextManager(config={"agents_md_sources": []})
        messages = mgr.build_conversation_context("hi", "s")
        assert all(m.role != MessageRole.SYSTEM for m in messages)


# ---------------------------------------------------------------------------
# _discover_agents_md_paths (ISSUE-012)
# ---------------------------------------------------------------------------

class TestDiscoverAgentsMdPaths:
    def test_returns_agents_md_from_distribution(self, tmp_path: "Path") -> None:
        p = tmp_path / "AGENTS.md"
        p.write_text("# Test", encoding="utf-8")

        mock_file = MagicMock()
        mock_file.name = "AGENTS.md"
        mock_file.locate.return_value.resolve.return_value = p

        mock_dist = MagicMock()
        mock_dist.files = [mock_file]

        with patch(
            "ovos_agentic_loop.context.agents_md.importlib.metadata.distributions",
            return_value=[mock_dist],
        ):
            paths = _discover_agents_md_paths()
        assert str(p) in paths

    def test_skips_nonexistent_file(self, tmp_path: "Path") -> None:
        p = tmp_path / "MISSING.md"  # not created on disk

        mock_file = MagicMock()
        mock_file.name = "AGENTS.md"
        mock_file.locate.return_value.resolve.return_value = p

        mock_dist = MagicMock()
        mock_dist.files = [mock_file]

        with patch(
            "ovos_agentic_loop.context.agents_md.importlib.metadata.distributions",
            return_value=[mock_dist],
        ):
            paths = _discover_agents_md_paths()
        assert paths == []

    def test_returns_empty_on_metadata_error(self) -> None:
        with patch(
            "ovos_agentic_loop.context.agents_md.importlib.metadata.distributions",
            side_effect=Exception("no dists"),
        ):
            paths = _discover_agents_md_paths()
        assert paths == []
