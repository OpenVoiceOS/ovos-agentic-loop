# AUDIT — ovos-agentic-loop

## Known Issues

### ISSUE-001 — ReActLoopEngine brain injection
`react.py:116` — `_load_brain()` silently returns `None` on any exception. A misconfigured `brain` plugin ID will only surface as "Error: no brain configured." at call time, not at init.

### ISSUE-002 — SkillMDToolBox brain not injected from loop engine
`toolbox.py` — `SkillMDToolBox.set_brain()` must be called explicitly; the owning `ReActLoopEngine` does not automatically share its brain with sub-toolboxes. Requires manual wiring in persona config or a future auto-injection pass.

### ISSUE-003 — No async support
All `continue_chat` paths are synchronous. Long tool chains block the event loop. Future work: async variant of `AgenticLoopEngine`.

### ISSUE-004 — ReAct prompt is English-only
`react.py:28` — The ReAct system prompt is hard-coded in English. No i18n support.
