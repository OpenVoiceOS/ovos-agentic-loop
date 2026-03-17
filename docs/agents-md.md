# AgentsMDContextManager — AGENTS.md Context Plugin

## Overview

`AgentsMDContextManager` (`ovos_agentic_loop/context/agents_md.py:67`) is an `AgentContextManager` plugin that loads one or more `AGENTS.md` files, filters their sections by heading, and assembles a system prompt for an LLM agent.

The same `AGENTS.md` that governs Claude Code at dev-time can govern a runtime LLM agent at inference time — both consumers read the same document.

**Entry point**: `opm.agents.memory` → `ovos-agents-md-context-plugin`

---

## AGENTS.md Loading and Parsing

### Path Resolution — `_resolve_paths()` — `agents_md.py:126`

1. Reads `config["agents_md_sources"]` (default: `["auto"]`).
2. If `"auto"` is in the list, calls `_discover_agents_md_paths()` — `agents_md.py:38` — which walks all installed distributions looking for files named `AGENTS.md` using `importlib.metadata.distributions()`.
3. Any explicit paths in `agents_md_sources` (other than `"auto"`) that exist on disk are appended.
4. `self._extra_paths` (constructor argument) are appended.
5. Deduplicates while preserving order — `agents_md.py:143`.

### Section Parsing — `_parse_sections(text)` — `agents_md.py:10`

Parses a markdown document into a dict `{heading_title: section_body}`:

- Recognises headings at any level (`#`, `##`, `###`, etc.) using `re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)` — `agents_md.py:26`.
- Section body is the text from the end of the heading line to the start of the next heading (at any level), stripped.
- Returns `{}` for an empty document.

Note: the parser does **not** respect heading hierarchy. A `###` subsection terminates its parent `##` section's body. This is intentional: it enables fine-grained `include_sections` filtering.

---

## Section Filtering — `include_sections`

`include_sections` is a list of heading substrings — `agents_md.py:114`. Filtering is case-insensitive substring matching — `agents_md.py:177`:

```python
if not any(inc.lower() in heading.lower() for inc in self.include_sections):
    continue
```

If `include_sections` is empty, **all sections** are included.

Example: `include_sections: ["Universal Rules", "OpenVoiceOS Workspace"]` includes any heading containing either substring (e.g. `"## Universal Rules"`, `"### Universal Rules — Python"`).

---

## System Prompt Construction — `_build_system_prompt()` — `agents_md.py:156`

1. Starts with `system_prompt_prefix` if non-empty — `agents_md.py:164`.
2. For each resolved path, reads the file, parses sections, filters, and formats each section as `"## {heading}\n\n{body}"`.
3. Joins all parts with `"\n\n"`.
4. Caches the result in `self._system_prompt` — `agents_md.py:191`.

The result is available via the `system_prompt` property — `agents_md.py:185`. It is lazily computed on first access and cached until `invalidate_cache()` is called — `agents_md.py:195`.

---

## History Management

`AgentsMDContextManager` maintains a flat in-memory list `self._history: List[AgentMessage]` — `agents_md.py:105`.

**Difference from OPM base interface**: The OPM `AgentContextManager` abstract interface (`agents.py:63`) requires `get_history(session_id: str)` and `update_history(new_messages: List[AgentMessage], session_id: str)` — both with a `session_id` parameter. `AgentsMDContextManager` implements `get_history()` and `update_history(message)` **without** `session_id` — `agents_md.py:203, 212`. This is a **signature mismatch** (see AUDIT ISSUE-006).

| Method | OPM Signature | Implemented Signature |
| :--- | :--- | :--- |
| `get_history` | `(self, session_id: str)` | `(self)` |
| `update_history` | `(self, new_messages: List[AgentMessage], session_id: str)` | `(self, message: AgentMessage)` |
| `build_conversation_context` | `(self, utterance: str, session_id: str)` | `(self, utterance: str, lang: Optional[str] = None)` |

---

## Building the Final Context — `build_conversation_context()` — `agents_md.py:221`

Produces the full `List[AgentMessage]` for an LLM call:

```
[SYSTEM: assembled system prompt]   (only if system_prompt is non-empty)
[...self._history...]
[USER: utterance]
```

- System message is prepended only when the assembled prompt is non-empty — `agents_md.py:243`.
- History is included in chronological order (insertion order of `update_history` calls).
- The current utterance is always the **last** message with `MessageRole.USER` — `agents_md.py:246`.
- The `lang` parameter is accepted but unused — `agents_md.py:231`.

---

## Config Reference

| Key | Type | Default | Source |
| :--- | :--- | :--- | :--- |
| `agents_md_sources` | List[str] | `["auto"]` | `agents_md.py:133` |
| `include_sections` | List[str] | `[]` | `agents_md.py:115` |
| `system_prompt_prefix` | str | `""` | `agents_md.py:120` |

---

## Example Config

```json
{
  "ovos-agents-md-context-plugin": {
    "agents_md_sources": ["auto", "/home/user/project/AGENTS.md"],
    "include_sections": ["Universal Rules", "OpenVoiceOS Workspace"],
    "system_prompt_prefix": "You are an OVOS developer assistant. Follow all rules strictly."
  }
}
```

With this config, the system prompt will be:

```
You are an OVOS developer assistant. Follow all rules strictly.

## Universal Rules

<content of Universal Rules section>

## OpenVoiceOS Workspace

<content of OpenVoiceOS Workspace section>
```

---

## Cache Invalidation

Call `invalidate_cache()` — `agents_md.py:195` — to force the system prompt to be rebuilt on the next access. Useful when an `AGENTS.md` file on disk has been updated during a long-running agent session.
