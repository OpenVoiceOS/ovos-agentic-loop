# SKILL.md Format and Discovery

## Overview

A `SKILL.md` file is a **dual-purpose document**: it describes a skill's API for Claude Code (dev-time) and simultaneously serves as the system prompt for a sub-LLM when a `SkillMDToolBox` tool is invoked at runtime.

Any Python package can ship a `SKILL.md`. Once installed, it is automatically discovered by `SkillMDLoader` and converted into an `AgentTool` accessible to any `ReActLoopEngine` that has `SkillMDToolBox` registered.

---

## Format Specification

A `SKILL.md` must begin with a YAML-style frontmatter block:

```markdown
---
name: <slug-friendly skill name>
description: <one-line tool description shown to the LLM>
---

<body: any markdown content>
```

### Frontmatter

Parsed by `_parse_skill_md()` (`ovos_agentic_loop/skills/loader.py:30`). The parser:

- Requires a leading `---\n` delimiter and a closing `---\n` delimiter (regex `^---\s*\n(.*?)\n---\s*\n(.*)` with DOTALL).
- Extracts `key: value` pairs matching `^(\w[\w-]*):\s*(.+)$` (`loader.py:61`). Nested YAML is **not** supported.
- Returns `None` (skips the file) if either `name` or `description` is missing (`loader.py:67`).

Required fields:

| Field | Description |
| :--- | :--- |
| `name` | Skill identifier. Slugified to form the tool name seen by the LLM. |
| `description` | One-line description. Used verbatim as `AgentTool.description`. |

Optional fields: Any additional `key: value` pairs are stored in `SkillMDEntry.raw_frontmatter` (`loader.py:27`), but not used by the toolbox.

### Body

Everything after the second `---` delimiter, stripped of leading/trailing whitespace (`loader.py:56`). The body:

- Becomes the `MessageRole.SYSTEM` prompt in the sub-LLM call (`toolbox.py:122`).
- Can contain any markdown: headings, lists, code blocks, instructions, examples.
- Should describe the skill's capabilities, constraints, and output format.

---

## Authoring Guide

A well-formed SKILL.md for a skill that searches the web:

```markdown
---
name: web-search
description: Use when the agent needs to search the web for current or factual information.
author: OpenVoiceOS
version: 1.0
---

## Instructions

You are a web search assistant. Given a task, search the web and return concise, factual answers.

- Always cite source URLs.
- Limit responses to the most relevant facts.
- If multiple conflicting sources exist, note the discrepancy.
- Prefer recent sources (within the last 2 years) for time-sensitive topics.

## Output Format

Respond with:
1. A direct answer to the question.
2. A list of source URLs.
```

### Authoring Rules

1. `name` must be unique across all installed SKILL.md files. Duplicate names produce duplicate tools. No error is raised.
2. `description` is shown directly to the LLM. Make it specific enough that the agent selects the right tool.
3. The body replaces the entire system prompt of the sub-LLM. Do not rely on any outer persona context being visible.
4. Tool name is derived from `name` via `_slugify()` (`toolbox.py:12`). Characters outside `[a-z0-9]` become underscores. `"web-search"` → `"web_search"`.

---

## Discovery Strategies

`SkillMDLoader` (`loader.py:143`) runs two discovery strategies in order, then appends any `extra_paths` from config:

### 1. Entry-point discovery: `_discover_via_entry_points()` (`loader.py:79`)

Packages register SKILL.md paths under the `opm.agents.skill_md` entry-point group:

```toml
[project.entry-points."opm.agents.skill_md"]
my-web-search = "my_package.skills:SKILL_MD_PATH"
```

Where `SKILL_MD_PATH` is a module attribute holding the absolute path string.

Fallback: if loading the attribute fails, the raw `ep.value` string is treated as a path.

This is the **recommended method** for packages that want zero-ambiguity discovery.

### 2. Package-data scan: `_discover_via_package_data()` (`loader.py:114`)

Walks all installed distributions' recorded file manifests looking for files named exactly `SKILL.md` (`loader.py:134`). Uses `importlib.metadata.distributions()`. Resolves paths via `f.locate().resolve()`. Files that do not exist on disk are silently skipped.

This is the **zero-config fallback**: if a package ships `SKILL.md` in its package root and it appears in the distribution RECORD, it will be found automatically.

### 3. Extra paths: `SkillMDLoader.extra_paths` (`loader.py:174`)

Set at construction time or via `config["extra_skill_md_paths"]` in `SkillMDToolBox` (`toolbox.py:84`). Useful for development, testing, or ad-hoc skill injection.

### Deduplication

All three strategies are concatenated and deduplicated (preserving order) before parsing (`loader.py:183`).

---

## Parsed Result: `SkillMDEntry`

`SkillMDEntry` (`loader.py:9`) is a dataclass with fields:

| Field | Type | Description |
| :--- | :--- | :--- |
| `name` | str | From frontmatter |
| `description` | str | From frontmatter |
| `body` | str | Markdown body (the system prompt) |
| `path` | str | Absolute path to the source file |
| `raw_frontmatter` | Dict[str, str] | All parsed frontmatter key-value pairs |

---

## Runtime Tool Invocation

When the agent loop calls a SKILL.md-backed tool:

1. `SkillMDToolBox.discover_tools()` (`toolbox.py:128`) has already built one `AgentTool` per `SkillMDEntry`.
2. The `tool_call` lambda captures the `SkillMDEntry` via default-argument closure (`toolbox.py:144`).
3. `_invoke_skill(entry, args)` (`toolbox.py:96`) assembles `[SYSTEM: entry.body, USER: args.task ...]` and calls `self._brain.continue_chat(messages)`.
4. Returns `SkillCallOutput(result=response.content, skill_used=entry.name)`.

The brain's response is the sub-LLM's answer guided entirely by the SKILL.md body. The outer loop sees it as a string observation via `str(result)` (`react.py:193`).

---

## Shipping a SKILL.md in Your Package

1. Create `your_package/SKILL.md` with valid frontmatter.
2. Add to `pyproject.toml`:
   ```toml
   [project.entry-points."opm.agents.skill_md"]
   my-skill-name = "your_package:SKILL_MD_PATH"
   ```
3. In `your_package/__init__.py` (or any importable module):
   ```python
   import os
   SKILL_MD_PATH: str = os.path.join(os.path.dirname(__file__), "SKILL.md")
   ```
4. Ensure `SKILL.md` is listed in `package_data` or `MANIFEST.in` so it appears in the distribution RECORD.

After `pip install -e .`, `SkillMDLoader` will discover and parse it.

---
[← Toolboxes](toolboxes.md) · [Home](../README.md) · [AGENTS.md →](agents-md.md)
