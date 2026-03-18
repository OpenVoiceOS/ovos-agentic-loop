# AUDIT — ovos-agentic-loop

Evidence-based audit with source citations. Severity: Critical / Major / Minor / Nitpick.
Status: **OPEN** / **ACCEPTED** (won't fix in current scope).

---

## Security

### ISSUE-005 — ShellToolBox: shell injection risk (Major) MITIGATED

**File**: `ovos_agentic_loop/tools/shell.py:85`

`subprocess.run(args.command, shell=True, ...)` passes the LLM-generated command string directly to `/bin/sh`. No sandboxing or argument-level escaping.

**Mitigations in place**:
- `allow_shell` defaults to `False` — `shell.py:63`. Must be explicitly enabled.
- `allowed_commands` config key: when non-empty, only commands whose first word matches a listed prefix are executed; others return `returncode=-1` without any subprocess call — `shell.py:85`.
- Documented in `docs/toolboxes.md`.

**Residual risk**: `allowed_commands` matches by string prefix, not argument-level validation. A sufficiently permissive allowlist (e.g. `["bash"]`) still exposes arbitrary execution. Operators must keep the allowlist tight.

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

### ISSUE-018 — No async support (Accepted)

**File**: `ovos_agentic_loop/base.py`, `ovos_agentic_loop/react.py`

All `continue_chat` paths are synchronous. Long tool chains block the calling thread. Tracked as `SUG-008`.
