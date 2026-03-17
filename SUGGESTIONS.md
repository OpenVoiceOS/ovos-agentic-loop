# SUGGESTIONS — ovos-agentic-loop

Agent proposals for future improvements. Each item references the relevant AUDIT issue where applicable.

---

## High Priority

### SUG-001 — Fix AgentContextManager interface compliance (fixes AUDIT ISSUE-001)

`AgentsMDContextManager` must implement the OPM abstract signatures:

- `get_history(self, session_id: str) -> List[AgentMessage]`
- `update_history(self, new_messages: List[AgentMessage], session_id: str) -> None`
- `build_conversation_context(self, utterance: str, session_id: str) -> List[AgentMessage]`

Migrate `self._history: List[AgentMessage]` to `self._history: Dict[str, List[AgentMessage]]` keyed by `session_id`. This is a breaking change; plan as a `0.2.0` minor bump.

---

### SUG-002 — Auto brain sharing in ReActLoopEngine (fixes AUDIT ISSUE-002)

In `ReActLoopEngine.continue_chat()` (or in `load_toolboxes()`), add a pass over `self.toolboxes` that calls `tb.set_brain(self.brain)` on any toolbox that has a `set_brain` method:

```python
for tb in self.toolboxes:
    if hasattr(tb, "set_brain") and self.brain is not None:
        tb.set_brain(self.brain)
```

This removes the need for manual wiring in persona configs and closes a major usability gap.

---

### SUG-003 — Add path sandboxing to FileSystemToolBox (fixes AUDIT ISSUE-006)

Add a `root_path` config key. All read, write, list, search, and find operations should resolve the target path and verify it is under `root_path`. Raise a `PermissionError` (surfaced as a tool error observation) if the path escapes the sandbox.

---

### SUG-004 — Add command allowlist to ShellToolBox (fixes AUDIT ISSUE-005)

Add an `allowed_commands` config key (list of allowed command prefixes). If set, `run_command` rejects any command whose first token is not in the list. This reduces shell injection risk without fully disabling the tool.

---

## Medium Priority

### SUG-005 — Add logging to silent failure paths (fixes AUDIT ISSUE-003, ISSUE-004)

Replace bare `except Exception: pass` in `_load_brain()` and `_load_toolboxes_from_config()` with `LOG.warning(...)` calls from `ovos_utils.log`. This makes misconfiguration visible in logs without changing behaviour.

---

### SUG-006 — Async variant: AsyncAgenticLoopEngine

Add `async_continue_chat()` to `AgenticLoopEngine` and `ReActLoopEngine`. Tool calls could be awaited in parallel when multiple independent observations are possible. This requires the brain `ChatEngine` to also offer an async interface.

An alternative is to run the synchronous loop in a thread pool executor: `await loop.run_in_executor(None, engine.continue_chat, messages)`. Document this pattern in FAQ.md as a temporary workaround.

---

### SUG-007 — Streaming support for outer loop

Hook `ReActLoopEngine.continue_chat` into `ChatEngine.stream_sentences()`. Yield partial sentences from each brain call as they arrive, with clear markers for tool calls vs final answer tokens. This enables TTS to begin speaking before the full answer is ready.

---

### SUG-008 — SkillMDLoader caching (fixes AUDIT ISSUE-018)

Add a `cached: bool = True` config key to `SkillMDToolBox`. When enabled, `SkillMDLoader.load()` result is cached after the first call. Provide a `reload()` method to invalidate the cache. This reduces repeated filesystem and metadata scans in long-running processes.

---

### SUG-009 — PlanExecuteLoopEngine

Add `ovos_agentic_loop/plan_execute.py` implementing a two-phase loop:

1. **Plan phase**: Ask the brain to produce a numbered step list for the given task.
2. **Execute phase**: Execute each step using tools, collecting results.
3. **Synthesis phase**: Ask the brain to synthesise a final answer from all step results.

This pattern is more predictable than ReAct for multi-step procedural tasks (e.g. "install X, run tests, report results").

---

### SUG-010 — Tool result caching

Add a `CachingToolBoxWrapper` that wraps any `ToolBox` and memoizes tool calls by `(tool_name, args_hash)` within a session. Useful for `web_search` and `read_file` where the same query might appear in multiple loop iterations.

---

### SUG-011 — Better FINAL_ANSWER parsing (fixes AUDIT ISSUE-009)

Replace the greedy regex `r"Action Input:\s*(\{.*?\})"` with a JSON bracket-counting parser that correctly handles nested JSON objects. This eliminates false negatives when the LLM produces valid but deeply nested `Action Input` JSON.

---

### SUG-012 — SKILL.md authoring helper utility

Add a `register_skill_md(path: str)` function in a `cli.py` module that generates and prints the correct `pyproject.toml` snippet for registering a SKILL.md. Lowers the barrier to shipping skill packages.

---

### SUG-013 — Multilingual ReAct prompt

Parameterise `_REACT_SYSTEM_PROMPT` by language. Add a `prompt_lang` config key to `ReActLoopEngine`. For unsupported languages, fall back to English. This addresses AUDIT ISSUE-007 and improves quality for non-English-primary models.

---

### SUG-014 — Tool call history / tracing

Expose the intermediate loop steps (thoughts, actions, observations) as metadata on the returned `AgentMessage`. This enables persona services to log or display the reasoning trace. A `trace: List[dict]` field could be added to `AgentMessage` or returned as a separate object.
