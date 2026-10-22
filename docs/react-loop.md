# ReActLoopEngine: Deep Dive

## Overview

`ReActLoopEngine` (`ovos_agentic_loop/react.py:92`) implements the **ReAct** (Reason + Act) algorithm: a structured prompt loop where an LLM alternates between reasoning steps (*Thought*), tool invocations (*Action*), and tool results (*Observation*) until it produces a *FINAL_ANSWER*.

`ReActLoopEngine` is a concrete subclass of `AgenticLoopEngine` (`ovos_agentic_loop/base.py:8`), which is itself a `ChatEngine` subclass. From the outside, callers receive exactly one `AgentMessage` back.

`ReActLoopEnginePlugin` (`ovos_agentic_loop/factory.py:8`) is the OPM-registered wrapper: a zero-body subclass of `ReActLoopEngine` that exists solely so `pyproject.toml` can point to a stable fully-qualified name.

---

## The ReAct Algorithm: Step by Step

1. **Collect tool schemas**: `_collect_tool_schemas()` (`react.py:159`)
   - Iterates `self.toolboxes`, calls `tb.tool_json_list` on each.
   - Returns a flat list of dicts: `[{name, description, argument_schema, output_schema}, ...]`.

2. **Prepend ReAct system message** (`react.py:227`)
   - If at least one tool schema exists, builds a `MessageRole.SYSTEM` message using `_build_react_system(tool_schemas)` (`react.py:38`).
   - Inserts it as the first message in the local `loop_messages` list (does **not** mutate the caller's list).
   - If no tools are available, no ReAct prompt is injected and the brain behaves as a plain ChatEngine.

3. **Iteration loop** (`react.py:235`)
   - Calls `self.brain.continue_chat(loop_messages, ...)` to get a text response.
   - Checks for `FINAL_ANSWER:` via `_extract_final_answer()` (`react.py:76`). If found, returns immediately.
   - Tries to parse an action via `_parse_action()` (`react.py:54`). If no action found, returns the raw text as the final answer.
   - Calls `_call_tool(tool_name, args)` (`react.py:175`), which dispatches to the first matching toolbox.
   - Appends the assistant's reasoning step and a `Observation: <result>` user message to `loop_messages`.
   - Repeats up to `max_iterations` times.

4. **Max-iterations fallback** (`react.py:262`)
   - When the loop exhausts without a final answer, appends a final user message demanding `FINAL_ANSWER:`.
   - Calls the brain once more; extracts the answer or uses the raw response as-is.

---

## Prompt Format

### ReAct System Prompt

Defined in `_REACT_SYSTEM_PROMPT` (`react.py:15`). Injected verbatim with `{tool_schemas}` replaced by `json.dumps(tool_schemas, indent=2)`:

```
You have access to tools. On each turn you MUST choose one of:

1. Use a tool:
   Thought: <reason about what to do>
   Action: <tool_name>
   Action Input: <JSON object matching the tool's argument schema>

2. Give the final answer (only when you have enough information):
   Thought: <reason>
   FINAL_ANSWER: <your answer to the user>

Available tools (JSON schema):
[...JSON array...]

Rules:
- Never skip the Thought line.
- Action Input MUST be valid JSON.
- Call only ONE tool per turn.
- After receiving an Observation, continue reasoning.
- Use FINAL_ANSWER: only once, as the last step.
```

The `{tool_schemas}` placeholder is replaced by `_build_react_system()` (`react.py:38`).

### Tool Schema Format

Each schema dict is produced by the OPM `ToolBox.tool_json_list` property (`agent_tools.py:290`):

```json
{
  "name": "web_search",
  "description": "Search the web...",
  "argument_schema": { ...pydantic JSON schema... },
  "output_schema": { ...pydantic JSON schema... }
}
```

### Message Sequence During a Loop Iteration

```
[SYSTEM]  <ReAct instructions + all tool schemas>
[USER]    <original conversation history>
[USER]    <latest user question>
  --- iteration 1 ---
[ASSISTANT] Thought: I need to search.\nAction: web_search\nAction Input: {"query":"..."}
[USER]    Observation: {"results": [...], "query": "..."}
  --- iteration 2 ---
[ASSISTANT] Thought: I have the answer.\nFINAL_ANSWER: <answer>
```

Observation messages are appended as `MessageRole.USER` (`react.py:257`). This is standard ReAct notation; the LLM distinguishes observation from user input by the `Observation:` prefix.

---

## Action / Observation Parsing

### `_parse_action(text)`: `react.py:54`

Uses regex to extract:
- `Action:\s*(\S+)` → `tool_name`
- `Action Input:\s*(\{.*?\})` (DOTALL) → JSON string → `args_dict`

Returns `(tool_name, args_dict)` or `None` if either regex fails or JSON is invalid.

Edge cases:
- Multi-line JSON in Action Input is supported (DOTALL flag).
- Invalid JSON silently returns `None`, causing the raw text to be treated as final answer.
- No `Thought:` enforcement in parsing. The regex only looks for `Action:` / `Action Input:` lines.

### `_extract_final_answer(text)`: `react.py:76`

Simple string search for `FINAL_ANSWER:` (constant `_FINAL_ANSWER_TOKEN`, `react.py:11`). Returns everything after the token, stripped. Returns `None` if not found.

---

## Tool Dispatch

`_call_tool(tool_name, args)` (`react.py:175`):

1. Iterates `self.toolboxes` in registration order.
2. Calls `tb.get_tool(tool_name)`. Returns `None` if not found.
3. On first match, calls `tb.call_tool(tool_name, args)` (OPM `ToolBox` method with full Pydantic validation).
4. Converts result to string via `str(result)`.
5. If no toolbox owns the tool, returns `"Error: tool '<name>' not found."`.

The error string is fed back to the LLM as an `Observation`, giving the brain a chance to recover. Exceptions during tool execution are caught per-toolbox; the loop continues to the next toolbox before falling back to the error string.

---

## Brain Injection

The inner LLM (brain) is a `ChatEngine` instance. Two loading paths:

1. **Explicit injection**: `set_brain(brain)` (`react.py:134`). Used in tests and when the caller holds a pre-configured brain instance.
2. **Config-driven**: `_load_brain()` (`react.py:143`). Called lazily on first `brain` property access (`react.py:130`). Reads `config["brain"]` as an OPM plugin ID, calls `load_chat_plugin(brain_id, config=self.config.get(brain_id, {}))`.

If loading fails for any reason, `brain` property returns `None`. `continue_chat` returns `AgentMessage(role=ASSISTANT, content="Error: no brain configured.")` (`react.py:221`).

---

## FINAL_ANSWER Detection

Detection is performed **before** action parsing in each iteration (`react.py:242`). This ensures that if the LLM includes both a `FINAL_ANSWER:` and an `Action:` in the same response (malformed output), the final answer takes priority.

---

## Config Reference

| Key | Type | Default | Source |
| :--- | :--- | :--- | :--- |
| `brain` | str | `""` | `react.py:150` |
| `max_iterations` | int | `10` | `react.py:125` |
| `toolboxes` | List[str] | `[]` | `base.py:58` |
| `<brain-id>` | dict | `{}` | forwarded on load (`react.py:155`) |
| `<toolbox-id>` | dict | `{}` | forwarded on load (`base.py:65`) |

---

## No-Tools Behaviour

If `self.toolboxes` is empty, `_collect_tool_schemas()` returns `[]`. The ReAct system prompt is **not** prepended (`react.py:228`). The first brain call either produces a plain response (no action parsing, returned as-is) or a FINAL_ANSWER. The loop will exit on the first iteration in the common case.

---
[← Native Tool-Call](native-toolcall-loop.md) · [Home](../README.md) · [Toolboxes →](toolboxes.md)
