# AUDIT — ovos-agentic-loop

Evidence-based audit with source citations. Severity: Critical / Major / Minor / Nitpick.

---

## Interface Compliance

### ISSUE-001 — AgentContextManager method signature mismatch (Major)

**File**: `ovos_agentic_loop/context/agents_md.py:203, 212, 221`

The OPM `AgentContextManager` abstract base (`agents.py:63`) declares:

```python
def get_history(self, session_id: str) -> List[AgentMessage]: ...
def update_history(self, new_messages: List[AgentMessage], session_id: str): ...
def build_conversation_context(self, utterance: str, session_id: str) -> List[AgentMessage]: ...
```

`AgentsMDContextManager` implements:

```python
def get_history(self) -> List[AgentMessage]: ...               # missing session_id
def update_history(self, message: AgentMessage) -> None: ...   # wrong signature (single msg, no session_id)
def build_conversation_context(self, utterance: str, lang: Optional[str] = None) -> List[AgentMessage]: ...  # session_id replaced by lang
```

**Impact**: Any caller that follows the OPM interface and passes `session_id` will trigger a `TypeError`. The class is not a valid implementation of the OPM abstract base in the strict sense. Multi-session support is absent.

**Mitigation**: Currently, if `ovos-persona` calls `build_conversation_context(utterance, session_id=...)` as a keyword argument, Python will bind `session_id` to `lang` only if the keyword names match — they do not, so a `TypeError` will result.

---

### ISSUE-002 — SkillMDToolBox brain not auto-injected (Major)

**File**: `ovos_agentic_loop/skills/toolbox.py:87`

`SkillMDToolBox.set_brain()` must be called explicitly before any tool invocation. `ReActLoopEngine` does not automatically share its own brain instance with sub-toolboxes. Any persona config that includes `ovos-skill-md-toolbox` without manual wiring will result in a `RuntimeError: brain ChatEngine is not configured` at call time — `toolbox.py:113`.

**Mitigation**: None currently. Must be wired manually.

---

### ISSUE-003 — ReActLoopEngine brain load is fully silent (Minor)

**File**: `ovos_agentic_loop/react.py:143`

`_load_brain()` catches all exceptions with bare `except Exception` and returns `None` silently. A misconfigured or missing `brain` plugin ID surfaces only as `"Error: no brain configured."` at `continue_chat` call time — `react.py:221`. No log message is emitted.

**Impact**: Difficult to diagnose brain loading failures in production.

---

### ISSUE-004 — _load_toolboxes_from_config is fully silent (Minor)

**File**: `ovos_agentic_loop/base.py:68`

Toolbox loading failures are silently swallowed with `except Exception: pass`. No warning or log is emitted. A misconfigured toolbox ID will result in `self.toolboxes` missing that entry with no indication.

---

### ISSUE-005 — ShellToolBox: shell injection risk (Major)

**File**: `ovos_agentic_loop/tools/shell.py:75`

`subprocess.run(args.command, shell=True, ...)` passes the LLM-generated command string directly to `/bin/sh`. There is no input validation, command allowlist, or sandboxing. A compromised or adversarially prompted LLM could execute arbitrary commands.

**Guidance**: `allow_shell=False` in production deployments where the LLM is not fully trusted. Consider a command allowlist or `shell=False` with explicit argument parsing as a future improvement.

---

### ISSUE-006 — FileSystemToolBox: path traversal risk (Minor)

**File**: `ovos_agentic_loop/tools/filesystem.py:123`

`_read_file` accepts any absolute or relative path and resolves it via `Path(args.path)`. There is no path restriction (e.g. sandbox root). An agent could read `/etc/passwd`, `.ssh/id_rsa`, or any world-readable file.

**Guidance**: Add a `root_path` config key to restrict all operations to a subtree.

---

### ISSUE-007 — ReAct system prompt is English-only (Minor)

**File**: `ovos_agentic_loop/react.py:15`

`_REACT_SYSTEM_PROMPT` is a hard-coded English string. The `lang` parameter passed to `continue_chat` is forwarded to the brain but does not influence the system prompt language. LLMs may respond in the user's language regardless, but the structured output format instructions (`Thought:`, `Action:`, `Action Input:`, `FINAL_ANSWER:`) are only in English.

---

### ISSUE-008 — Observation role uses MessageRole.USER (Nitpick)

**File**: `ovos_agentic_loop/react.py:257`

Tool observations are injected as `MessageRole.USER` messages with an `Observation: ` prefix. Some LLM providers support a dedicated `tool` role; using `USER` may cause confusion in chat histories when reviewed or replayed.

---

### ISSUE-009 — _parse_action regex: greedy Action Input match (Minor)

**File**: `ovos_agentic_loop/react.py:65`

`r"Action Input:\s*(\{.*?\})"` with DOTALL uses a non-greedy match for the JSON object. If the LLM produces nested JSON with closing `}` characters before the outermost `}`, the regex may capture an incomplete JSON fragment, causing `json.JSONDecodeError` and returning `None`. The LLM's response would then be treated as a final answer rather than an action.

---

## Test Coverage Gaps

### ISSUE-010 — _discover_via_entry_points not tested with mocked metadata (Minor)

**File**: `test/test_loader.py`

`_discover_via_entry_points()` — `loader.py:79` — is not exercised by any test. The entry-point fallback path (raw `ep.value` as path) — `loader.py:108` — is also untested.

---

### ISSUE-011 — _discover_via_package_data not tested (Minor)

**File**: `test/test_loader.py`

`_discover_via_package_data()` — `loader.py:114` — is never directly tested. Only `extra_paths` loading is covered. The package-data scan involving `importlib.metadata.distributions()` is exercised only indirectly and only returns real-world results (not mocked).

---

### ISSUE-012 — AgentsMDContextManager auto discovery not tested (Minor)

**File**: `test/test_agents_md.py`

`_discover_agents_md_paths()` — `agents_md.py:38` — is not covered by any test. The `"auto"` source path triggers the distribution scan, which is expensive and environment-dependent. A mock-based test is needed.

---

### ISSUE-013 — ReActLoopEngine _load_toolboxes_from_config not tested (Minor)

**File**: `test/test_base.py`

`_load_toolboxes_from_config()` — `base.py:50` — is only tested indirectly. The OPM `load_toolbox_plugin` path is not exercised because OPM is not available in the test environment. Import failure branch is covered by the `ImportError` catch — `base.py:70` — but not explicitly asserted.

---

### ISSUE-014 — ReActLoopEngine _load_brain OPM path not tested (Minor)

**File**: `test/test_react.py`

All tests use `set_brain()`. The `_load_brain()` OPM path — `react.py:143` — is not tested. Missing `brain` key, OPM unavailable, and OPM returning `None` are not covered.

---

### ISSUE-015 — ToolBox bus protocol not tested (Minor)

**File**: `test/`

`ToolBox.bind()`, `handle_discover()`, and `handle_call()` are OPM base methods. The agentic-loop test suite has no test that verifies the bus event protocol (discovery response format, call-result format). This is partial coverage: the bus path exists but the format contract is unverified.

---

## Type Annotation Issues

### ISSUE-016 — AgenticLoopEngine.toolboxes typed as List[Any] (Nitpick)

**File**: `ovos_agentic_loop/base.py:35`

`self.toolboxes: List[Any]` avoids a circular import with `ToolBox`. Should be `List["ToolBox"]` or use `TYPE_CHECKING` — weakens static analysis on toolbox usage.

---

## Known Limitations

### ISSUE-003 (repeated) — No async support

**File**: `ovos_agentic_loop/base.py`, `ovos_agentic_loop/react.py`

All `continue_chat` paths are synchronous. Long tool chains (especially `ShellToolBox` or `WebSearchToolBox`) block the calling thread for the full duration. In an asyncio event loop, this stalls other coroutines.

---

### ISSUE-017 — No session isolation in AgentsMDContextManager

**File**: `ovos_agentic_loop/context/agents_md.py:105`

`self._history` is a single flat list shared across all sessions. Multiple concurrent callers will have their messages interleaved. This is a consequence of ISSUE-001.

---

### ISSUE-018 — SkillMDLoader does not cache parsed entries

**File**: `ovos_agentic_loop/skills/loader.py:196`

`SkillMDLoader.load()` re-parses all files on every call. `SkillMDToolBox.discover_tools()` calls `self._loader.load()` — `toolbox.py:136` — on every invocation (triggered by `ToolBox.refresh_tools()` and initial construction). For environments with many installed SKILL.md files, this is unnecessarily expensive.
