# MAINTENANCE_REPORT — ovos-agentic-loop

## 2026-03-17 — P0/P1 security and correctness fixes

- **AI Model**: claude-sonnet-4-6
- **Actions Taken**:
  - Fixed `AgentsMDContextManager` ABC compliance (per-session history, correct method signatures)
  - Fixed `_extract_json_object` balanced-brace parser replacing broken non-greedy regex in `react.py`
  - Flipped `ShellToolBox.allow_shell` default to `False` (secure by default)
  - Added `root_path` sandbox + `_safe_path()` path-traversal prevention in `FileSystemToolBox`
  - Added `_inject_brain_into_toolboxes()` to `base.py`; wired into all 7 loop engines
  - Replaced silent `except: pass` with `LOG.warning()` in `base.py`
  - Updated all affected tests; 149 tests, 88% coverage
- **Oversight**: Code review findings by human; fixes implemented by AI

## 2026-03-17 — Initial implementation

- **AI Model**: claude-sonnet-4-6
- **Actions Taken**:
  - Created repo from scratch implementing the plan: AgenticLoopEngine base, ReActLoopEngine, SkillMDLoader, SkillMDToolBox, AgentsMDContextManager
  - Also removed `AGENT_LOOP` / `AgenticLoopEngine` from `ovos-plugin-manager` (cleanup of premature OPM PR)
  - 49 unit tests, all passing
- **Oversight**: Plan reviewed and approved by human before implementation

---

## 2026-03-17 — Comprehensive documentation audit

- **AI Model**: claude-sonnet-4-6
- **Actions Taken**:
  - Read all Python source files, all test files, `pyproject.toml`, existing docs, and OPM template files (`agent_tools.py`, `agents.py`) for cross-reference.
  - Read `ovos-persona` source (`solvers.py`, `memory.py`) to understand how plugins are consumed.
  - **Rewrote** `docs/index.md`: full architecture overview with component table, OPM discovery, persona integration, message bus event flow, complete configuration reference, quick-start example.
  - **Created** `docs/react-loop.md`: ReAct algorithm step-by-step with source citations, prompt format, action/observation parsing, tool dispatch, brain injection, FINAL_ANSWER detection, config reference.
  - **Created** `docs/toolboxes.md`: per-toolbox reference for all 5 toolboxes with argument/output tables, config keys, security notes.
  - **Created** `docs/skill-md.md`: SKILL.md format spec (frontmatter, body), discovery strategies (entry points + package data scan), authoring guide, runtime invocation flow, packaging instructions.
  - **Created** `docs/agents-md.md`: path resolution, section parsing, `include_sections` filtering, system prompt construction, history management, signature mismatch analysis, cache invalidation.
  - **Created** `docs/opm-integration.md`: entry point groups, persona loading path, bus protocol, toolbox and SKILL.md registration guide, factory pattern.
  - **Rewrote** `AUDIT.md`: 18 issues identified across interface compliance, security, test coverage gaps, type annotation issues, and known limitations; all with `file:line` citations.
  - **Rewrote** `SUGGESTIONS.md`: 14 proposals covering interface fixes, security hardening, async support, streaming, caching, and new engine types.
  - **Updated** `MAINTENANCE_REPORT.md`: this entry.
- **Key findings**:
  - `AgentsMDContextManager` has a critical interface mismatch with the OPM `AgentContextManager` abstract base: `session_id` is absent from all three abstract methods (AUDIT ISSUE-001). This will cause `TypeError` if a standard persona service calls it.
  - `SkillMDToolBox` requires manual brain injection; the owning `ReActLoopEngine` does not auto-share its brain (AUDIT ISSUE-002).
  - `ShellToolBox` uses `shell=True` with no input sanitisation (AUDIT ISSUE-005).
  - `FileSystemToolBox` has no path sandboxing (AUDIT ISSUE-006).
  - Test coverage of discovery mechanisms (`_discover_via_entry_points`, `_discover_via_package_data`) is absent.
- **Oversight**: Documentation-only task; no code changes made.
