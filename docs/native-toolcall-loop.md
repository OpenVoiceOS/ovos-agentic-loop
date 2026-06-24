# NativeToolCallEngine — Deep Dive

## Overview

`NativeToolCallEngine` (`ovos_agentic_loop/native_toolcall.py`) drives a tool loop
using the brain's **native** function-calling instead of ReAct's text protocol. It
hands the toolboxes to the brain via `continue_chat(..., tools=...)` and reads back
**structured** `AgentMessage.tool_calls`, executes each through the existing
`ToolBox` machinery, and feeds the results back as `MessageRole.TOOL` messages until
the brain answers without requesting a tool.

It is a subclass of `ReActLoopEngine` (and therefore `AgenticLoopEngine` →
`ChatEngine`), reusing its brain wiring, toolbox loading, schema collection and tool
execution. Only the loop in `continue_chat` differs.

`NativeToolCallEnginePlugin` (`ovos_agentic_loop/factory.py`) is the OPM-registered
wrapper. Entry point: **`ovos-native-toolcall-loop`** (group `opm.agents.chat`).

## Native vs. ReAct — and the fallback

| | NativeToolCallEngine | ReActLoopEngine |
| --- | --- | --- |
| How the model calls a tool | provider-native `tool_calls` | `Action:`/`Action Input:` text, regex-parsed |
| Tool result fed back as | `MessageRole.TOOL` message | `Observation:` text in a user message |
| Brain requirement | `supports_tools = True` | any text ChatEngine |
| Reliability / token cost | higher / lower (provider parses) | lower / higher (prompt + parse) |

**Fallback:** if the configured brain does not advertise `supports_tools`,
`continue_chat` transparently delegates to the inherited ReAct text loop
(`super().continue_chat(...)`). So this engine works with **any** brain — it just
uses the better path when the brain supports it.

## The loop

1. **No brain** → return an error `AgentMessage`.
2. **Brain lacks `supports_tools`** → `return super().continue_chat(...)` (ReAct).
3. Otherwise iterate up to `max_iterations`:
   - `resp = brain.continue_chat(loop_messages, tools=self.toolboxes)` — the
     `ToolBox` objects are passed straight through; the brain normalizes them with
     `ToolBox.normalize_tools` to its provider's tool format.
   - If `resp.tool_calls` is empty → return `AgentMessage(ASSISTANT, resp.content)`.
   - Else append the assistant turn (carrying `tool_calls`) **first**, then one
     `MessageRole.TOOL` message per `ToolCall` (with `tool_call_id`/`name`),
     executing via the inherited `_call_tool`. The assistant-before-tool ordering
     is the invariant providers require.
4. **Max iterations** → one final tool-free `continue_chat(..., tools=None)` to
   force a text answer.

## Config

| Key | Meaning | Default |
| --- | --- | --- |
| `brain` | ChatEngine plugin id used as the inner LLM (must set `supports_tools` to use native calls) | — |
| `toolboxes` | ToolBox plugin ids to load | — |
| `max_iterations` | Max tool-call cycles before forcing an answer | 10 |

## Streaming

Tool calls are driven through the non-streaming `continue_chat`. Token/sentence
streaming applies only to the terminal answer; structured `tool_calls` are not
streamed.

## Example

See [`examples/native_toolcall_persona.json`](../examples/native_toolcall_persona.json)
and [`examples/native_toolcall_persona.py`](../examples/native_toolcall_persona.py).
