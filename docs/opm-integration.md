# OPM Plugin Integration

## Entry Point Groups

This package uses three OPM entry-point groups:

| Group | OPM Function | Purpose |
| :--- | :--- | :--- |
| `opm.agents.chat` | `find_chat_plugins()` / `load_chat_plugin()` | LLM chat engines (includes agentic loops) |
| `opm.agents.toolbox` | `find_toolbox_plugin()` / `load_toolbox_plugin()` | Tool collections for agent loops |
| `opm.agents.memory` | (context manager discovery) | Conversation context builders |
| `opm.agents.skill_md` | `importlib.metadata.entry_points()` | SKILL.md path registration (used by SkillMDLoader) |

---

## How ovos-persona Loads These Plugins

`ovos-persona` (`ovos_persona/solvers.py`) calls `get_utterance_handler_plugins()` — `solvers.py:22` — which aggregates results from multiple OPM `find_*` functions. `find_chat_plugins()` is included, which returns all `opm.agents.chat` entry points. `ReActLoopEnginePlugin` appears in this map when installed.

`QuestionSolversService.load_plugins()` — `solvers.py:43` — instantiates each plugin with `plug_class(config=config)`. The config is looked up by plugin ID from the persona config dict.

At chat time, `QuestionSolversService.chat_completion()` — `solvers.py:71` — calls `module.continue_chat(messages, session_id=..., lang=..., units=...)` for any `ChatEngine` instance. `ReActLoopEnginePlugin` is a `ChatEngine` subclass, so this dispatch path is used directly. The persona does not know or care that the engine runs an internal loop.

---

## ToolBox → Persona Message Bus Protocol

`ToolBox.bind(bus)` — OPM `agent_tools.py:89` — registers two handlers:

### Discovery

- **Trigger**: `ovos.persona.tools.discover`
- **Handler**: `ToolBox.handle_discover()` — `agent_tools.py:112`
- **Response**: `message.response({"tools": self.tool_json_list, "toolbox_id": self.toolbox_id})`

`tool_json_list` is a list of dicts with JSON Schema for each tool's arguments and outputs — `agent_tools.py:290`. A persona service can broadcast `ovos.persona.tools.discover` and collect schemas from all bound toolboxes.

### Tool Call

- **Trigger**: `ovos.persona.tools.<toolbox_id>.call`
- **Message data**: `{"name": "tool_name", "kwargs": {...}}`
- **Handler**: `ToolBox.handle_call()` — `agent_tools.py:128`
- **Response (success)**: `{"result": result.model_dump(), "toolbox_id": "..."}`
- **Response (error)**: `{"error": "ExceptionType: message", "toolbox_id": "..."}`

Note: `ReActLoopEngine` in this package calls tools **directly** via `tb.call_tool()` and does **not** use the message bus. The bus protocol is available to any external consumer (e.g. a persona service that wants to call tools over the bus from a separate process).

---

## How to Register a New ToolBox

1. Create a class inheriting from `ToolBox` (OPM `agent_tools.py:56`).
2. Set a unique `toolbox_id` class attribute.
3. Implement `discover_tools() -> List[AgentTool]`.
4. Register in `pyproject.toml`:

```toml
[project.entry-points."opm.agents.toolbox"]
my-custom-tools = "my_package.toolboxes:MyToolBox"
```

5. The toolbox can then be referenced by its entry point name in any `ReActLoopEngine` config:

```json
{
  "ovos-react-loop": {
    "toolboxes": ["my-custom-tools"],
    "my-custom-tools": { "option": "value" }
  }
}
```

`AgenticLoopEngine._load_toolboxes_from_config()` — `base.py:50` — calls `load_toolbox_plugin(tid, config=self.config.get(tid, {}))` for each entry in `config["toolboxes"]`.

---

## How to Register a New SKILL.md Package

Any package can expose a `SKILL.md` as a tool by adding an entry point under `opm.agents.skill_md`:

```toml
[project.entry-points."opm.agents.skill_md"]
my-skill-name = "my_package:SKILL_MD_PATH"
```

In `my_package/__init__.py`:

```python
import os
SKILL_MD_PATH: str = os.path.join(os.path.dirname(__file__), "SKILL.md")
```

After install, `SkillMDLoader._discover_via_entry_points()` — `loader.py:79` — will find and parse the file. No further configuration is needed.

Alternatively, include the file in the package data so it is listed in the distribution RECORD — `_discover_via_package_data()` — `loader.py:114` — will find it automatically without an entry point.

---

## ReActLoopEnginePlugin — OPM Factory Pattern

`ReActLoopEnginePlugin` (`factory.py:8`) is a zero-body subclass of `ReActLoopEngine`. This pattern is used throughout OVOS: the entry point target must be a stable class reference in a dedicated factory module. The actual implementation lives in `react.py`. If the implementation class name changes, only `factory.py` needs updating; `pyproject.toml` is unaffected.

---

## AgentsMDContextManager — Memory Plugin

`AgentsMDContextManager` is registered under `opm.agents.memory`. The OPM group name mirrors the `AgentContextManager` base class. Persona services that support this group can load context managers by plugin ID and call `build_conversation_context(utterance, lang)` to receive the assembled message list.

Cross-reference: `ovos-persona` uses `BasicShortTermMemory` (`ovos_persona/memory.py:6`) as its default context manager. `AgentsMDContextManager` can replace or supplement this for agents that need AGENTS.md-driven context.
