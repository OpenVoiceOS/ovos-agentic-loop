# AUDIT — ovos-agentic-loop

Evidence-based audit with source citations. Severity: Critical / Major / Minor / Nitpick.
Status: **RESOLVED** / **OPEN** / **ACCEPTED** (won't fix in current scope).

---

## Interface Compliance

### ISSUE-001 — AgentContextManager method signatures ✅ RESOLVED

**File**: `ovos_agentic_loop/context/agents_md.py:211`

Signatures now match OPM `AgentContextManager` ABC:
- `get_history(self, session_id: str) -> List[AgentMessage]`
- `update_history(self, new_messages: List[AgentMessage], session_id: str) -> None`
- `build_conversation_context(self, utterance: str, session_id: str) -> List[AgentMessage]`

Per-session history stored in `self._sessions: Dict[str, List[AgentMessage]]` — `agents_md.py:112`.

---

### ISSUE-002 — SkillMDToolBox brain not auto-injected ✅ RESOLVED

**File**: `ovos_agentic_loop/react.py:179`

`brain` property now calls `_inject_brain_into_toolboxes` after lazy-loading via OPM — `react.py:182`. Both the `set_brain()` explicit path and the lazy OPM path propagate the brain to all registered toolboxes.

---

### ISSUE-003 — ReActLoopEngine brain load silent ✅ RESOLVED

**File**: `ovos_agentic_loop/react.py:209`

`_load_brain()` now emits `LOG.warning(f"ReActLoopEngine: failed to load brain '{brain_id}': {exc}")` on failure — `react.py:212`.

---

### ISSUE-004 — _load_toolboxes_from_config silent ✅ RESOLVED

**File**: `ovos_agentic_loop/base.py:104`

`LOG.warning(f"AgenticLoopEngine: failed to load toolbox '{tid}': {exc}")` emitted on each toolbox load failure — `base.py:105`.

---

### ISSUE-005 — ShellToolBox: shell injection risk (Major) OPEN / ACCEPTED

**File**: `ovos_agentic_loop/tools/shell.py:75`

`subprocess.run(args.command, shell=True, ...)` passes the LLM-generated command string directly to `/bin/sh`. No input validation, command allowlist, or sandboxing.

**Mitigations in place**:
- `allow_shell` defaults to `False` — `shell.py:41`. Must be explicitly enabled.
- Documented in README Security Notes and `docs/toolboxes.md`.

**Accepted risk**: A command allowlist would be useful but is out of scope for `0.1.0`. Tracked as `SUG-003`.

---

### ISSUE-006 — FileSystemToolBox: path traversal ✅ RESOLVED

**File**: `ovos_agentic_loop/tools/filesystem.py:63`

`_safe_path(requested)` resolves all paths relative to `root_path` and rejects any path that escapes the sandbox — `filesystem.py:63`. Escaping attempts return an error string without touching the filesystem.

---

### ISSUE-007 — ReAct system prompt is English-only (Minor) OPEN

**File**: `ovos_agentic_loop/react.py:15`

`_REACT_SYSTEM_PROMPT` is a hard-coded English string. The structured output format tokens (`Thought:`, `Action:`, `FINAL_ANSWER:`) are English-only. Most capable LLMs comply regardless of user language, but strict multilingual compliance is not guaranteed.

**Mitigation**: None. Tracked as `SUG-007`.

---

### ISSUE-008 — Observation role uses MessageRole.USER (Nitpick) OPEN

**File**: `ovos_agentic_loop/react.py:310`

Tool observations are injected as `MessageRole.USER` with an `Observation: ` prefix. The OPM `MessageRole` enum does not include a dedicated `tool` role. Using `USER` is the correct pragmatic choice given current OPM capabilities.

**Accepted**: No action needed until OPM exposes a `tool` role.

---

### ISSUE-009 — Action Input regex truncates nested JSON ✅ RESOLVED

**File**: `ovos_agentic_loop/react.py:54`

Replaced greedy/non-greedy regex with a balanced-brace parser `_extract_json_object(text, start)` that correctly handles nested JSON objects — `react.py:54`. Tested by `TestParseAction` in `test/test_react.py`.

---

## Test Coverage Gaps

### ISSUE-010 — _discover_via_entry_points not tested (Minor) OPEN

**File**: `test/test_loader.py`

The entry-point fallback path in `loader.py:79` has no mock-based test. Tracking only; acceptable for `0.1.0`.

---

### ISSUE-011 — _discover_via_package_data not tested (Minor) OPEN

**File**: `test/test_loader.py`

`_discover_via_package_data()` — `loader.py:114` — not directly tested. Acceptable for `0.1.0`.

---

### ISSUE-012 — AgentsMDContextManager auto discovery not tested (Minor) OPEN

**File**: `test/test_agents_md.py`

`_discover_agents_md_paths()` not covered by any test. Acceptable for `0.1.0`.

---

### ISSUE-013 — _load_toolboxes_from_config OPM path not tested (Minor) OPEN

**File**: `test/test_base.py`

OPM plugin path in `base.py:93` not exercised. Acceptable for `0.1.0`.

---

### ISSUE-014 — _load_brain OPM path not tested (Minor) OPEN

**File**: `test/test_react.py`

All tests use `set_brain()`. The `_load_brain()` OPM path — `react.py:207` — not tested. Acceptable for `0.1.0`.

---

### ISSUE-015 — ToolBox bus protocol not tested (Minor) OPEN

**File**: `test/`

`ToolBox.bind()`, `handle_discover()`, `handle_call()` bus event format contracts are untested. OPM-level concern; acceptable for `0.1.0`.

---

## Type Annotation Issues

### ISSUE-016 — AgenticLoopEngine.toolboxes typed as List[Any] (Nitpick) OPEN

**File**: `ovos_agentic_loop/base.py:36`

`self.toolboxes: List[Any]` avoids a circular import with `ToolBox`. Should be `List["ToolBox"]` or use `TYPE_CHECKING`. Tracked as `SUG-008`.

---

## Known Limitations

### ISSUE-017 — No session isolation in AgentsMDContextManager ✅ RESOLVED

**File**: `ovos_agentic_loop/context/agents_md.py:112`

Per-session history stored in `self._sessions: Dict[str, List[AgentMessage]]` — each session isolated. Consequence of ISSUE-001 fix.

---

### ISSUE-018 — No async support (Accepted)

**File**: `ovos_agentic_loop/base.py`, `ovos_agentic_loop/react.py`

All `continue_chat` paths are synchronous. Long tool chains block the calling thread. Tracked as `SUG-009`.

---

### ISSUE-019 — SkillMDLoader does not cache parsed entries (Minor) OPEN

**File**: `ovos_agentic_loop/skills/loader.py`

`load()` re-parses all SKILL.md files on every call. Acceptable for `0.1.0`; tracked as `SUG-010`.
