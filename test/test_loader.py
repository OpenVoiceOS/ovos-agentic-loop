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

"""Unit tests for SkillMDLoader and _parse_skill_md."""
import os
import textwrap
import time
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from ovos_agentic_loop.skills.loader import (
    SkillMDLoader,
    _discover_via_entry_points,
    _discover_via_package_data,
    _parse_skill_md,
)


VALID_SKILL_MD = textwrap.dedent("""\
    ---
    name: web-search
    description: Use when the agent needs to search the web for current information.
    ---

    ## Instructions

    Search the web and return concise, factual answers.
    Cite sources when possible.
""")

MISSING_NAME_MD = textwrap.dedent("""\
    ---
    description: Something without a name.
    ---
    body here
""")

NO_FRONTMATTER_MD = "Just a regular markdown file without frontmatter."


class TestParseSkillMd:
    def test_valid_file(self, tmp_path: "Path") -> None:
        p = tmp_path / "SKILL.md"
        p.write_text(VALID_SKILL_MD, encoding="utf-8")
        entry = _parse_skill_md(str(p))
        assert entry is not None
        assert entry.name == "web-search"
        assert "current information" in entry.description
        assert "Instructions" in entry.body
        assert entry.path == str(p)
        assert entry.raw_frontmatter["name"] == "web-search"

    def test_missing_name_returns_none(self, tmp_path: "Path") -> None:
        p = tmp_path / "SKILL.md"
        p.write_text(MISSING_NAME_MD, encoding="utf-8")
        assert _parse_skill_md(str(p)) is None

    def test_no_frontmatter_returns_none(self, tmp_path: "Path") -> None:
        p = tmp_path / "SKILL.md"
        p.write_text(NO_FRONTMATTER_MD, encoding="utf-8")
        assert _parse_skill_md(str(p)) is None

    def test_nonexistent_file_returns_none(self) -> None:
        assert _parse_skill_md("/does/not/exist/SKILL.md") is None


class TestSkillMDLoader:
    def test_load_from_extra_paths(self, tmp_path: "Path") -> None:
        p = tmp_path / "SKILL.md"
        p.write_text(VALID_SKILL_MD, encoding="utf-8")
        loader = SkillMDLoader(extra_paths=[str(p)])
        entries = loader.load()
        names = [e.name for e in entries]
        assert "web-search" in names

    def test_load_skips_invalid(self, tmp_path: "Path") -> None:
        valid = tmp_path / "SKILL.md"
        valid.write_text(VALID_SKILL_MD, encoding="utf-8")
        invalid = tmp_path / "BAD_SKILL.md"
        invalid.write_text(NO_FRONTMATTER_MD, encoding="utf-8")
        loader = SkillMDLoader(extra_paths=[str(valid), str(invalid)])
        entries = loader.load()
        # BAD_SKILL.md has no frontmatter — must not be in results.
        names = [e.name for e in entries]
        assert "web-search" in names
        # No entry should have a name parsed from NO_FRONTMATTER_MD text.
        assert all(e.path != str(invalid) for e in entries)

    def test_discover_paths_deduplicated(self, tmp_path: "Path") -> None:
        p = tmp_path / "SKILL.md"
        p.write_text(VALID_SKILL_MD, encoding="utf-8")
        loader = SkillMDLoader(extra_paths=[str(p), str(p)])
        paths = loader.discover_paths()
        assert paths.count(str(p)) == 1

    def test_empty_loader_returns_empty(self) -> None:
        loader = SkillMDLoader(extra_paths=[])
        # Package-data and entry-point discovery may find nothing in a bare env.
        entries = loader.load()
        assert isinstance(entries, list)

    def test_load_caches_result(self, tmp_path: "Path") -> None:
        p = tmp_path / "SKILL.md"
        p.write_text(VALID_SKILL_MD, encoding="utf-8")
        loader = SkillMDLoader(extra_paths=[str(p)])
        first = loader.load()
        second = loader.load()
        assert first == second

    def test_load_invalidated_when_mtime_changes(self, tmp_path: "Path") -> None:
        p = tmp_path / "SKILL.md"
        p.write_text(VALID_SKILL_MD, encoding="utf-8")
        loader = SkillMDLoader(extra_paths=[str(p)])
        first = loader.load()
        # Touch the file to update mtime.
        time.sleep(0.01)
        p.write_text(VALID_SKILL_MD.replace("web-search", "updated-search"), encoding="utf-8")
        second = loader.load()
        assert second[0].name == "updated-search"
        assert first[0].name != second[0].name

    def test_invalidate_cache_forces_reparse(self, tmp_path: "Path") -> None:
        p = tmp_path / "SKILL.md"
        p.write_text(VALID_SKILL_MD, encoding="utf-8")
        loader = SkillMDLoader(extra_paths=[str(p)])
        loader.load()
        loader.invalidate_cache()
        assert loader._cache is None
        entries = loader.load()
        assert len(entries) == 1


# ---------------------------------------------------------------------------
# _discover_via_entry_points (ISSUE-010)
# ---------------------------------------------------------------------------

class TestDiscoverViaEntryPoints:
    def test_returns_path_from_valid_entry_point(self, tmp_path: "Path") -> None:
        p = tmp_path / "SKILL.md"
        p.write_text(VALID_SKILL_MD, encoding="utf-8")

        ep = MagicMock()
        ep.load.return_value = str(p)

        with patch("ovos_agentic_loop.skills.loader.importlib.metadata.entry_points",
                   return_value=[ep]):
            paths = _discover_via_entry_points()
        assert str(p) in paths

    def test_falls_back_to_ep_value_on_load_error(self, tmp_path: "Path") -> None:
        p = tmp_path / "SKILL.md"
        p.write_text(VALID_SKILL_MD, encoding="utf-8")

        ep = MagicMock()
        ep.load.side_effect = ImportError("broken")
        ep.value = str(p)

        with patch("ovos_agentic_loop.skills.loader.importlib.metadata.entry_points",
                   return_value=[ep]):
            paths = _discover_via_entry_points()
        assert str(p) in paths

    def test_ignores_nonexistent_path(self, tmp_path: "Path") -> None:
        ep = MagicMock()
        ep.load.return_value = str(tmp_path / "missing.md")

        with patch("ovos_agentic_loop.skills.loader.importlib.metadata.entry_points",
                   return_value=[ep]):
            paths = _discover_via_entry_points()
        assert paths == []

    def test_returns_empty_on_metadata_error(self) -> None:
        with patch("ovos_agentic_loop.skills.loader.importlib.metadata.entry_points",
                   side_effect=Exception("no metadata")):
            paths = _discover_via_entry_points()
        assert paths == []


# ---------------------------------------------------------------------------
# _discover_via_package_data (ISSUE-011)
# ---------------------------------------------------------------------------

class TestDiscoverViaPackageData:
    def test_returns_skill_md_paths_from_distributions(self, tmp_path: "Path") -> None:
        p = tmp_path / "SKILL.md"
        p.write_text(VALID_SKILL_MD, encoding="utf-8")

        mock_file = MagicMock()
        mock_file.name = "SKILL.md"
        mock_file.locate.return_value.resolve.return_value = p

        mock_dist = MagicMock()
        mock_dist.files = [mock_file]

        with patch("ovos_agentic_loop.skills.loader.importlib.metadata.distributions",
                   return_value=[mock_dist]):
            paths = _discover_via_package_data()
        assert str(p) in paths

    def test_skips_nonexistent_file(self, tmp_path: "Path") -> None:
        p = tmp_path / "NONEXISTENT.md"  # not created on disk

        mock_file = MagicMock()
        mock_file.name = "SKILL.md"
        mock_file.locate.return_value.resolve.return_value = p

        mock_dist = MagicMock()
        mock_dist.files = [mock_file]

        with patch("ovos_agentic_loop.skills.loader.importlib.metadata.distributions",
                   return_value=[mock_dist]):
            paths = _discover_via_package_data()
        assert paths == []

    def test_returns_empty_on_metadata_error(self) -> None:
        with patch("ovos_agentic_loop.skills.loader.importlib.metadata.distributions",
                   side_effect=Exception("broken")):
            paths = _discover_via_package_data()
        assert paths == []
