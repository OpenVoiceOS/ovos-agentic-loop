"""SkillMDLoader — discovers and parses SKILL.md files from installed packages."""
import importlib.metadata
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SkillMDEntry:
    """
    Parsed representation of a single SKILL.md file.

    Attributes:
        name: Skill name from frontmatter (e.g. ``"web-search"``).
        description: One-line description from frontmatter, used as the tool
            description shown to the LLM.
        body: Full markdown body (everything after the frontmatter block),
            used as the sub-LLM system prompt when invoking the skill.
        path: Absolute filesystem path to the source SKILL.md file.
        raw_frontmatter: All key→value pairs parsed from the YAML frontmatter.
    """
    name: str
    description: str
    body: str
    path: str
    raw_frontmatter: Dict[str, str] = field(default_factory=dict)


def _parse_skill_md(path: str) -> Optional[SkillMDEntry]:
    """
    Parse a SKILL.md file into a ``SkillMDEntry``.

    Expects YAML-style frontmatter delimited by ``---`` lines at the top of
    the file.  Only string-valued fields are extracted; the remainder of the
    file becomes the ``body``.

    Args:
        path: Absolute path to the SKILL.md file.

    Returns:
        ``SkillMDEntry`` if parsing succeeds, ``None`` otherwise.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return None

    # Split frontmatter from body.
    fm_pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)
    match = fm_pattern.match(text)
    if not match:
        return None

    fm_text, body = match.group(1), match.group(2).strip()

    # Parse simple ``key: value`` pairs — no nested YAML.
    raw: Dict[str, str] = {}
    for line in fm_text.splitlines():
        kv = re.match(r"^(\w[\w-]*):\s*(.+)$", line.strip())
        if kv:
            raw[kv.group(1).strip()] = kv.group(2).strip()

    name = raw.get("name", "")
    description = raw.get("description", "")
    if not name or not description:
        return None

    return SkillMDEntry(
        name=name,
        description=description,
        body=body,
        path=path,
        raw_frontmatter=raw,
    )


def _discover_via_entry_points() -> List[str]:
    """
    Find SKILL.md paths declared under the ``opm.agents.skill_md`` entry-point group.

    Packages that want explicit discovery should add an entry like::

        [project.entry-points."opm.agents.skill_md"]
        my-skill = "my_package:SKILL_MD_PATH"

    where the value is an importable attribute that holds the path string, or
    alternatively the entry-point ``value`` itself is used as a path if the
    attribute cannot be resolved.

    Returns:
        List of absolute SKILL.md file paths.
    """
    paths: List[str] = []
    try:
        eps = importlib.metadata.entry_points(group="opm.agents.skill_md")
    except Exception:  # noqa: BLE001
        return paths

    for ep in eps:
        try:
            value = ep.load()
            if isinstance(value, str) and os.path.isfile(value):
                paths.append(value)
        except Exception:  # noqa: BLE001 — skip broken entry points
            # Fallback: treat ep.value as a direct path string.
            candidate = ep.value
            if isinstance(candidate, str) and os.path.isfile(candidate):
                paths.append(candidate)
    return paths


def _discover_via_package_data() -> List[str]:
    """
    Scan installed package data for files named ``SKILL.md``.

    Iterates over all installed distributions and checks their recorded files
    for paths ending in ``SKILL.md``.

    Returns:
        List of absolute SKILL.md file paths that exist on disk.
    """
    paths: List[str] = []
    try:
        dists = importlib.metadata.distributions()
    except Exception:  # noqa: BLE001
        return paths

    for dist in dists:
        try:
            files = dist.files or []
            for f in files:
                if f.name == "SKILL.md":
                    abs_path = str(f.locate().resolve())
                    if os.path.isfile(abs_path):
                        paths.append(abs_path)
        except Exception:  # noqa: BLE001
            continue
    return paths


class SkillMDLoader:
    """
    Discovers all installed ``SKILL.md`` files and parses them into
    ``SkillMDEntry`` objects.

    Discovery order:

    1. **Entry points** — ``opm.agents.skill_md`` group (explicit, zero-ambiguity).
    2. **Package data scan** — walks every installed distribution looking for
       files named ``SKILL.md`` (zero-config fallback).

    Duplicate paths (same absolute path from both strategies) are deduplicated.

    Attributes:
        extra_paths: Additional filesystem paths to search, set at construction
            time.

    Usage::

        loader = SkillMDLoader(extra_paths=["/my/project/SKILL.md"])
        entries = loader.load()
    """

    def __init__(self, extra_paths: Optional[List[str]] = None) -> None:
        """
        Initialise the loader.

        Args:
            extra_paths: Optional list of additional SKILL.md file paths to
                include regardless of entry-point or package-data discovery.
        """
        self.extra_paths: List[str] = list(extra_paths or [])

    def discover_paths(self) -> List[str]:
        """
        Return deduplicated list of all SKILL.md file paths.

        Returns:
            Absolute paths, deduplicated while preserving order (entry-point
            paths first, then package-data paths, then extra_paths).
        """
        seen: set = set()
        result: List[str] = []
        for path in (
            _discover_via_entry_points()
            + _discover_via_package_data()
            + self.extra_paths
        ):
            if path not in seen:
                seen.add(path)
                result.append(path)
        return result

    def load(self) -> List[SkillMDEntry]:
        """
        Discover and parse all SKILL.md files.

        Returns:
            List of successfully parsed ``SkillMDEntry`` objects.  Files that
            fail to parse are silently skipped.
        """
        entries: List[SkillMDEntry] = []
        for path in self.discover_paths():
            entry = _parse_skill_md(path)
            if entry is not None:
                entries.append(entry)
        return entries
