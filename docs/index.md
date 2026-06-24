# ovos-agentic-loop — Architecture Overview

## What this package does

`ovos-agentic-loop` provides a plugin framework for **agentic (tool-using) LLM loops** within the OVOS ecosystem. It supplies:

1. An abstract base class (`AgenticLoopEngine`) for any agent-loop `ChatEngine` plugin.
2. Seven loop engines from agent literature: **Chain-of-Thought**, **ReAct**, **Plan-and-Execute**, **Reflexion**, **Self-Ask**, **CRITIC**, **Tree-of-Thoughts**.
3. Five `ToolBox` plugins: filesystem, shell, web-search, clock, and SKILL.md-backed tools.
4. `SkillMDLoader` and `SkillMDToolBox` for converting installed `SKILL.md` files into callable tools.
5. `AgentsMDContextManager` for assembling system prompts from installed `AGENTS.md` files.

All components integrate with `ovos-plugin-manager` (OPM) via entry points and are discovered at runtime by `ovos-persona` or any OPM consumer.

**Key architectural insight**: SKILL.md and AGENTS.md are dual-purpose documents — they govern Claude Code at dev-time and serve as tool descriptors / behavioural constraints for runtime LLM agents.

---

## Component Table

| Class | File | Entry Point Group | Entry Point ID |
| :--- | :--- | :--- | :--- |
| `AgenticLoopEngine` | `ovos_agentic_loop/base.py:8` | — (abstract base) | — |
| `ReActLoopEngine` | `ovos_agentic_loop/react.py:92` | — (concrete) | — |
| `ReActLoopEnginePlugin` | `ovos_agentic_loop/factory.py:8` | `opm.agents.chat` | `ovos-react-loop` |
| `NativeToolCallEngine` | `ovos_agentic_loop/native_toolcall.py` | — (concrete) | — |
| `NativeToolCallEnginePlugin` | `ovos_agentic_loop/factory.py` | `opm.agents.chat` | `ovos-native-toolcall-loop` |
| `PlanAndExecuteEngine` | `ovos_agentic_loop/plan_execute.py:108` | — (concrete) | — |
| `PlanAndExecuteEnginePlugin` | `ovos_agentic_loop/factory.py:27` | `opm.agents.chat` | `ovos-plan-execute-loop` |
| `ReflexionEngine` | `ovos_agentic_loop/reflexion.py:82` | — (concrete) | — |
| `ReflexionEnginePlugin` | `ovos_agentic_loop/factory.py:36` | `opm.agents.chat` | `ovos-reflexion-loop` |
| `SelfAskEngine` | `ovos_agentic_loop/self_ask.py:112` | — (concrete) | — |
| `SelfAskEnginePlugin` | `ovos_agentic_loop/factory.py:45` | `opm.agents.chat` | `ovos-self-ask-loop` |
| `ChainOfThoughtEngine` | `ovos_agentic_loop/chain_of_thought.py:68` | — (concrete) | — |
| `ChainOfThoughtEnginePlugin` | `ovos_agentic_loop/factory.py:54` | `opm.agents.chat` | `ovos-chain-of-thought-loop` |
| `CRITICEngine` | `ovos_agentic_loop/critic.py:92` | — (concrete) | — |
| `CRITICEnginePlugin` | `ovos_agentic_loop/factory.py:63` | `opm.agents.chat` | `ovos-critic-loop` |
| `TreeOfThoughtsEngine` | `ovos_agentic_loop/tree_of_thoughts.py:108` | — (concrete) | — |
| `TreeOfThoughtsEnginePlugin` | `ovos_agentic_loop/factory.py:72` | `opm.agents.chat` | `ovos-tree-of-thoughts-loop` |
| `SkillMDLoader` | `ovos_agentic_loop/skills/loader.py:143` | — | — |
| `SkillMDToolBox` | `ovos_agentic_loop/skills/toolbox.py:48` | `opm.agents.toolbox` | `ovos-skill-md-toolbox` |
| `FileSystemToolBox` | `ovos_agentic_loop/tools/filesystem.py:85` | `opm.agents.toolbox` | `ovos-filesystem-tools` |
| `ShellToolBox` | `ovos_agentic_loop/tools/shell.py:26` | `opm.agents.toolbox` | `ovos-shell-tools` |
| `WebSearchToolBox` | `ovos_agentic_loop/tools/web.py:25` | `opm.agents.toolbox` | `ovos-web-search-tools` |
| `ClockToolBox` | `ovos_agentic_loop/tools/clock.py:22` | `opm.agents.toolbox` | `ovos-clock-tools` |
| `AgentsMDContextManager` | `ovos_agentic_loop/context/agents_md.py:67` | `opm.agents.memory` | `ovos-agents-md-context-plugin` |

---

## OPM Plugin Discovery

Entry points in `pyproject.toml` (lines 26–37):

```toml
[project.entry-points."opm.agents.chat"]
ovos-react-loop = "ovos_agentic_loop.factory:ReActLoopEnginePlugin"

[project.entry-points."opm.agents.toolbox"]
ovos-skill-md-toolbox   = "ovos_agentic_loop.skills.toolbox:SkillMDToolBox"
ovos-filesystem-tools   = "ovos_agentic_loop.tools.filesystem:FileSystemToolBox"
ovos-shell-tools        = "ovos_agentic_loop.tools.shell:ShellToolBox"
ovos-web-search-tools   = "ovos_agentic_loop.tools.web:WebSearchToolBox"
ovos-clock-tools        = "ovos_agentic_loop.tools.clock:ClockToolBox"

[project.entry-points."opm.agents.memory"]
ovos-agents-md-context-plugin = "ovos_agentic_loop.context.agents_md:AgentsMDContextManager"
```

OPM uses `importlib.metadata.entry_points()` to discover classes at runtime. `opm.agents.chat` maps to `find_chat_plugins()` / `load_chat_plugin()`. `opm.agents.toolbox` maps to `find_toolbox_plugin()` / `load_toolbox_plugin()`.

---

## Integration with ovos-persona

`ovos-persona` (`ovos_persona/solvers.py:22`) loads all `ChatEngine` plugins via `find_chat_plugins()`. `ReActLoopEnginePlugin` is a `ChatEngine` subclass (via `AgenticLoopEngine → ChatEngine`), so `ovos-persona` treats it identically to any LLM plugin — it calls `continue_chat(messages, session_id, lang, units)` and receives one `AgentMessage` back.

All loop mechanics are **opaque to the caller**: tool selection, execution, observation injection, and iteration happen inside `ReActLoopEngine.continue_chat` — `ovos_agentic_loop/react.py:198`.

`ToolBox` plugins are loaded by `ReActLoopEngine._load_toolboxes_from_config` — `ovos_agentic_loop/base.py:50` — using `load_toolbox_plugin()` from OPM. They can also be injected directly via `AgenticLoopEngine.load_toolboxes()` — `ovos_agentic_loop/base.py:38`.

`AgentsMDContextManager` is an `AgentContextManager` subclass. It can be loaded by any persona service that supports the `opm.agents.memory` group and calls `build_conversation_context(utterance, lang)`.

---

## Message Bus Event Flow (ToolBox)

`ToolBox` (OPM `agent_tools.py:56`) registers two bus handlers when `bind(bus)` is called:

| Event | Direction | Payload |
| :--- | :--- | :--- |
| `ovos.persona.tools.discover` | → ToolBox | (empty) |
| `ovos.persona.tools.discover` (response) | ToolBox → | `{tools: [...schemas...], toolbox_id: "..."}` |
| `ovos.persona.tools.<toolbox_id>.call` | → ToolBox | `{name: "tool_name", kwargs: {...}}` |
| `ovos.persona.tools.<toolbox_id>.call` (response) | ToolBox → | `{result: {...model_dump...}, toolbox_id}` or `{error: "..."}` |

`ReActLoopEngine` calls tools **directly** (not over the bus) via `tb.call_tool(name, args)` — `ovos_agentic_loop/react.py:175`. Bus dispatch is a capability of the OPM `ToolBox` base class available to other consumers.

---

## Configuration Reference

### ReActLoopEnginePlugin (`ovos-react-loop`)

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `brain` | str | `""` | OPM plugin ID of the inner `ChatEngine` for LLM calls |
| `toolboxes` | List[str] | `[]` | OPM plugin IDs of `ToolBox` plugins to load |
| `max_iterations` | int | `10` | Maximum tool-call cycles per `continue_chat` invocation |
| `<brain-plugin-id>` | dict | `{}` | Config dict forwarded to the brain plugin on load |
| `<toolbox-plugin-id>` | dict | `{}` | Config dict forwarded to each toolbox plugin on load |

### FileSystemToolBox (`ovos-filesystem-tools`)

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `allow_write` | bool | `True` | If `False`, `write_file` returns `success=False` without touching disk |

### ShellToolBox (`ovos-shell-tools`)

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `allow_shell` | bool | `True` | If `False`, `run_command` returns an error without executing |
| `max_timeout` | int | `120` | Upper bound (seconds) for any requested command timeout |

### SkillMDToolBox (`ovos-skill-md-toolbox`)

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `extra_skill_md_paths` | List[str] | `[]` | Additional SKILL.md file paths beyond auto-discovery |

### AgentsMDContextManager (`ovos-agents-md-context-plugin`)

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `agents_md_sources` | List[str] | `["auto"]` | Paths to AGENTS.md files, or `["auto"]` for package-data discovery |
| `include_sections` | List[str] | `[]` | Heading substrings to include (empty = all sections) |
| `system_prompt_prefix` | str | `""` | Text prepended before assembled AGENTS.md content |

---

## Quick-Start Example

```python
from ovos_agentic_loop.react import ReActLoopEngine
from ovos_agentic_loop.tools import ClockToolBox, WebSearchToolBox
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

# Create engine; brain must be a pre-loaded ChatEngine instance or loadable by OPM.
engine = ReActLoopEngine(config={"max_iterations": 5})
engine.set_brain(my_chat_engine)  # inject an instantiated ChatEngine
engine.load_toolboxes([ClockToolBox(), WebSearchToolBox()])

messages = [AgentMessage(role=MessageRole.USER, content="What time is it in Tokyo?")]
response = engine.continue_chat(messages)
print(response.content)
```

Persona config snippet (OVOS JSON config):

```json
{
  "name": "OVOSDeveloperAgent",
  "solvers": ["ovos-react-loop"],
  "plugin-config": {
    "ovos-react-loop": {
      "brain": "ovos-llm-plugin",
      "max_iterations": 5,
      "toolboxes": ["ovos-clock-tools", "ovos-web-search-tools"]
    }
  }
}
```

---

## See Also

- `docs/loop-architectures.md` — All four loop engines: rationale, algorithm, when to use, comparison table
- `docs/react-loop.md` — ReAct algorithm deep dive with all source citations
- `docs/toolboxes.md` — Per-toolbox reference (args, outputs, config)
- `docs/skill-md.md` — SKILL.md format spec, discovery, and authoring guide
- `docs/agents-md.md` — AGENTS.md context manager internals
- `docs/opm-integration.md` — OPM entry point integration and plugin registration guide
