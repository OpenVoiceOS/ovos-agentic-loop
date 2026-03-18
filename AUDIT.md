# AUDIT — ovos-agentic-loop

Evidence-based audit with source citations. Severity: Critical / Major / Minor / Nitpick.
Status: **OPEN** / **ACCEPTED** (won't fix in current scope).

---

## Security

### ISSUE-005 — ShellToolBox: shell injection risk (Major) ACCEPTED

**File**: `ovos_agentic_loop/tools/shell.py:75`

`subprocess.run(args.command, shell=True, ...)` passes the LLM-generated command string directly to `/bin/sh`. No input validation, command allowlist, or sandboxing.

**Mitigations in place**:
- `allow_shell` defaults to `False` — `shell.py:41`. Must be explicitly enabled.
- Documented in README Security Notes and `docs/toolboxes.md`.

**Accepted risk**: A command allowlist would be useful but is out of scope for `0.1.0`. Tracked as `SUG-003`.

---

## Interface

### ISSUE-007 — ReAct system prompt is English-only (Minor) OPEN

**File**: `ovos_agentic_loop/react.py:15`

`_REACT_SYSTEM_PROMPT` is a hard-coded English string. The structured output format tokens (`Thought:`, `Action:`, `FINAL_ANSWER:`) are English-only. Most capable LLMs comply regardless of user language, but strict multilingual compliance is not guaranteed.

**Mitigation**: None. Tracked as `SUG-007`.

---

### ISSUE-008 — Observation role uses MessageRole.USER (Nitpick) ACCEPTED

**File**: `ovos_agentic_loop/react.py:310`

Tool observations are injected as `MessageRole.USER` with an `Observation: ` prefix. The OPM `MessageRole` enum does not include a dedicated `tool` role. Using `USER` is the correct pragmatic choice given current OPM capabilities.

**Accepted**: No action needed until OPM exposes a `tool` role.

---

## Known Limitations

### ISSUE-015 — ToolBox bus protocol not tested (Minor) ACCEPTED

**File**: `test/`

`ToolBox.bind()`, `handle_discover()`, `handle_call()` bus event format contracts are untested. OPM-level concern; acceptable for `0.1.0`.

---

### ISSUE-018 — No async support (Accepted)

**File**: `ovos_agentic_loop/base.py`, `ovos_agentic_loop/react.py`

All `continue_chat` paths are synchronous. Long tool chains block the calling thread. Tracked as `SUG-008`.
