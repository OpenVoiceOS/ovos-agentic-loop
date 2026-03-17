"""Unit tests for SkillMDLoader and _parse_skill_md."""
import os
import textwrap
from typing import List

import pytest

from ovos_agentic_loop.skills.loader import SkillMDLoader, _parse_skill_md


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
