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

"""Unit tests for the standard developer toolboxes."""
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ovos_agentic_loop.tools.clock import ClockToolBox, GetCurrentDatetimeArgs
from ovos_agentic_loop.tools.filesystem import (
    FileSystemToolBox,
    FindFilesArgs,
    ListDirectoryArgs,
    ReadFileArgs,
    SearchInFilesArgs,
    WriteFileArgs,
)
from ovos_agentic_loop.tools.shell import RunCommandArgs, ShellToolBox
from ovos_agentic_loop.tools.web import WebSearchArgs, WebSearchToolBox


# ---------------------------------------------------------------------------
# FileSystemToolBox
# ---------------------------------------------------------------------------

class TestFileSystemToolBox:
    """Tests for FileSystemToolBox."""

    def _box(self, tmp_path: Path, **cfg: Any) -> "FileSystemToolBox":
        """Create a sandboxed FileSystemToolBox rooted at tmp_path."""
        return FileSystemToolBox(config={"root_path": str(tmp_path), **cfg})

    def test_read_existing_file(self, tmp_path: Path) -> None:
        """read_file returns the file content."""
        f = tmp_path / "hello.txt"
        f.write_text("hello world", encoding="utf-8")
        result = self._box(tmp_path).call_tool("read_file", {"path": "hello.txt"})
        assert result.content == "hello world"
        assert "hello.txt" in result.path

    def test_read_missing_file(self, tmp_path: Path) -> None:
        """read_file returns an error message for a non-existent path."""
        result = self._box(tmp_path).call_tool("read_file", {"path": "nonexistent.txt"})
        assert "Error" in result.content

    def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        """Paths that escape the sandbox root are rejected."""
        result = self._box(tmp_path).call_tool("read_file", {"path": "../../etc/passwd"})
        assert "Access denied" in result.content

    def test_write_read_roundtrip(self, tmp_path: Path) -> None:
        """write_file creates the file; read_file retrieves its contents."""
        box = self._box(tmp_path)
        write_result = box.call_tool("write_file", {"path": "subdir/out.txt", "content": "round-trip"})
        assert write_result.success is True
        read_result = box.call_tool("read_file", {"path": "subdir/out.txt"})
        assert read_result.content == "round-trip"

    def test_write_blocked_when_disabled(self, tmp_path: Path) -> None:
        """write_file returns success=False when allow_write=False."""
        box = self._box(tmp_path, allow_write=False)
        result = box.call_tool("write_file", {"path": "nope.txt", "content": "x"})
        assert result.success is False
        assert not (tmp_path / "nope.txt").exists()

    def test_list_directory(self, tmp_path: Path) -> None:
        """list_directory returns files in the directory."""
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.txt").write_text("")
        result = self._box(tmp_path).call_tool("list_directory", {"path": "."})
        names = [Path(e).name for e in result.entries]
        assert "a.py" in names
        assert "b.txt" in names

    def test_list_directory_with_pattern(self, tmp_path: Path) -> None:
        """list_directory respects the glob pattern filter."""
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.txt").write_text("")
        result = self._box(tmp_path).call_tool("list_directory", {"path": ".", "pattern": "*.py"})
        names = [Path(e).name for e in result.entries]
        assert "a.py" in names
        assert "b.txt" not in names

    def test_search_in_files_finds_match(self, tmp_path: Path) -> None:
        """search_in_files returns matching lines."""
        (tmp_path / "code.py").write_text("def foo():\n    return 42\n")
        result = self._box(tmp_path).call_tool("search_in_files", {"pattern": "def foo", "path": "."})
        assert result.total >= 1
        assert any("def foo" in m["line"] for m in result.matches)

    def test_search_in_files_no_match(self, tmp_path: Path) -> None:
        """search_in_files returns empty list when no lines match."""
        (tmp_path / "code.py").write_text("hello world\n")
        result = self._box(tmp_path).call_tool("search_in_files", {"pattern": "zzznomatch", "path": "."})
        assert result.total == 0

    def test_find_files_glob(self, tmp_path: Path) -> None:
        """find_files returns matching file paths."""
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        result = self._box(tmp_path).call_tool("find_files", {"glob": "**/*.py", "path": "."})
        names = [Path(f).name for f in result.files]
        assert "a.py" in names
        assert "b.py" in names
        assert "c.txt" not in names

    def test_discover_tools_returns_five(self, tmp_path: Path) -> None:
        """discover_tools always returns 5 tools."""
        assert len(self._box(tmp_path).discover_tools()) == 5


# ---------------------------------------------------------------------------
# ShellToolBox
# ---------------------------------------------------------------------------

class TestShellToolBox:
    """Tests for ShellToolBox."""

    def test_shell_blocked_by_default(self) -> None:
        """run_command is disabled by default (allow_shell=False)."""
        box = ShellToolBox()
        result = box.call_tool("run_command", {"command": "echo hello"})
        assert result.success is False
        assert "disabled" in result.stderr.lower()

    def test_run_echo(self) -> None:
        """run_command captures stdout for a simple echo when allow_shell=True."""
        box = ShellToolBox(config={"allow_shell": True})
        result = box.call_tool("run_command", {"command": "echo hello"})
        assert result.success is True
        assert "hello" in result.stdout
        assert result.returncode == 0

    def test_run_failing_command(self) -> None:
        """run_command reports returncode != 0 for a failing command."""
        box = ShellToolBox(config={"allow_shell": True})
        result = box.call_tool("run_command", {"command": "exit 1", "cwd": "."})
        assert result.success is False
        assert result.returncode != 0

    def test_timeout_respected(self) -> None:
        """run_command times out when the command exceeds the requested timeout."""
        box = ShellToolBox(config={"allow_shell": True})
        result = box.call_tool("run_command", {"command": "sleep 60", "timeout": 1})
        assert result.success is False
        assert "timed out" in result.stderr.lower()

    def test_max_timeout_caps_requested_timeout(self) -> None:
        """max_timeout config caps any higher requested timeout."""
        box = ShellToolBox(config={"allow_shell": True, "max_timeout": 5})
        result = box.call_tool("run_command", {"command": "echo ok", "timeout": 999})
        assert result.success is True

    def test_shell_blocked_when_disabled(self) -> None:
        """run_command returns success=False when allow_shell=False."""
        box = ShellToolBox(config={"allow_shell": False})
        result = box.call_tool("run_command", {"command": "echo hello"})
        assert result.success is False
        assert "disabled" in result.stderr.lower()

    def test_discover_tools_returns_one(self) -> None:
        """discover_tools returns exactly one tool."""
        box = ShellToolBox()
        assert len(box.discover_tools()) == 1


# ---------------------------------------------------------------------------
# WebSearchToolBox
# ---------------------------------------------------------------------------

class TestWebSearchToolBox:
    """Tests for WebSearchToolBox."""

    def test_web_search_with_mock(self) -> None:
        """web_search returns structured results when DDGS is available."""
        fake_results = [
            {"title": "Result 1", "href": "https://example.com/1", "body": "Snippet 1"},
            {"title": "Result 2", "href": "https://example.com/2", "body": "Snippet 2"},
        ]
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = fake_results

        box = WebSearchToolBox()
        with patch.dict("sys.modules", {"duckduckgo_search": MagicMock(DDGS=MagicMock(return_value=mock_ddgs))}):
            result = box.call_tool("web_search", {"query": "ovos voice assistant", "max_results": 2})

        assert result.query == "ovos voice assistant"
        assert len(result.results) == 2
        assert result.results[0]["title"] == "Result 1"
        assert result.results[0]["url"] == "https://example.com/1"
        assert result.results[0]["snippet"] == "Snippet 1"

    def test_missing_package_graceful_error(self) -> None:
        """web_search returns a friendly message when duckduckgo_search is missing."""
        box = WebSearchToolBox()
        # Temporarily remove the module from sys.modules if present, then block import.
        saved = sys.modules.pop("duckduckgo_search", None)
        try:
            with patch.dict("sys.modules", {"duckduckgo_search": None}):  # type: ignore[dict-item]
                result = box.call_tool("web_search", {"query": "test"})
            assert len(result.results) == 1
            assert "not installed" in result.results[0]["title"].lower() or "package" in result.results[0]["snippet"].lower()
        finally:
            if saved is not None:
                sys.modules["duckduckgo_search"] = saved

    def test_discover_tools_returns_one(self) -> None:
        """discover_tools returns exactly one tool."""
        box = WebSearchToolBox()
        assert len(box.discover_tools()) == 1


# ---------------------------------------------------------------------------
# ClockToolBox
# ---------------------------------------------------------------------------

class TestClockToolBox:
    """Tests for ClockToolBox."""

    def test_returns_valid_iso_datetime(self) -> None:
        """get_current_datetime returns a parseable ISO datetime string."""
        from datetime import datetime
        box = ClockToolBox()
        result = box.call_tool("get_current_datetime", {})
        # Should not raise
        dt = datetime.fromisoformat(result.iso)
        assert dt is not None

    def test_date_field_nonempty(self) -> None:
        """date field is a non-empty YYYY-MM-DD string."""
        import re
        box = ClockToolBox()
        result = box.call_tool("get_current_datetime", {})
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", result.date)

    def test_time_field_nonempty(self) -> None:
        """time field is a non-empty HH:MM:SS string."""
        import re
        box = ClockToolBox()
        result = box.call_tool("get_current_datetime", {})
        assert re.match(r"^\d{2}:\d{2}:\d{2}$", result.time)

    def test_timezone_field_nonempty(self) -> None:
        """timezone field is non-empty."""
        box = ClockToolBox()
        result = box.call_tool("get_current_datetime", {})
        assert result.timezone

    def test_discover_tools_returns_one(self) -> None:
        """discover_tools returns exactly one tool."""
        box = ClockToolBox()
        assert len(box.discover_tools()) == 1
