# SUGGESTIONS — ovos-agentic-loop

Agent proposals for future improvements. Each item references the relevant AUDIT issue where applicable.
Status: ✅ DONE / OPEN / LOW-PRI

---

## High Priority

### SUG-001 — Fix AgentContextManager interface compliance ✅ DONE

`AgentsMDContextManager` now implements correct OPM ABC signatures with per-session `_sessions` dict. See AUDIT ISSUE-001.

---

### SUG-002 — Auto brain sharing in ReActLoopEngine ✅ DONE

`brain` property now calls `_inject_brain_into_toolboxes` after lazy-loading — `react.py:182`. Both explicit `set_brain()` and OPM-loaded paths propagate the brain. See AUDIT ISSUE-002.

---

### SUG-003 — ShellToolBox command allowlist (fixes AUDIT ISSUE-005)

Add a `allowed_commands` config key: list of permitted command prefixes (e.g. `["git", "ls", "cat"]`). Commands not matching any prefix return an error without execution. This allows `allow_shell: true` in controlled environments without full shell access.

```python
# Proposed API
config = {
    "allow_shell": True,
    "allowed_commands": ["git status", "git log", "ls"],
}
```

---

### SUG-004 — Add LOG.warning for silent brain load failure ✅ DONE

`_load_brain()` now emits `LOG.warning` on failure — `react.py:212`. See AUDIT ISSUE-003.

---

## Medium Priority

### SUG-005 — SkillMDLoader caching ✅ DONE

`SkillMDLoader.load()` now caches parsed entries and invalidates when `extra_paths` or any file mtime changes — `skills/loader.py:209`. `invalidate_cache()` forces a full re-parse.

---

### SUG-006 — TYPE_CHECKING guard for ToolBox in base.py ✅ DONE

`AgenticLoopEngine.toolboxes` is now typed as `List["ToolBox"]` via a `TYPE_CHECKING` guard — `base.py:17`. No runtime circular import.

---

### SUG-007 — Multilingual ReAct system prompt (fixes AUDIT ISSUE-007)

Accept a `system_prompt_lang` config key and ship translated prompt templates. Fall back to English when no translation is available. Low-priority given that capable LLMs comply with English instructions even in non-English conversations.

---

### SUG-008 — Async `continue_chat` variants (fixes AUDIT ISSUE-018)

Add `async def acontinue_chat(...)` to `AgenticLoopEngine` and `ReActLoopEngine`. Tool calls could run with `asyncio.to_thread` for CPU-bound operations. Long-running tools (web search, shell) would no longer block the event loop.

---

## Low Priority

### SUG-009 — Streaming output support

Add `stream_sentences(messages, ...)` to loop engines so partial answers can be fed to TTS incrementally. Requires the brain ChatEngine to support streaming — not all do.

---

### SUG-010 — Tree-of-Thoughts parallelism

`TreeOfThoughtsEngine` evaluates branches sequentially. Parallelize branch generation and scoring with `concurrent.futures.ThreadPoolExecutor` for a significant speed-up on capable hardware.

---

### SUG-011 — Per-loop token budget tracking

Add a `token_budget` config key. Accumulate estimated token counts across iterations and stop early when the budget is exceeded. Prevents runaway costs with pay-per-token LLM APIs.
