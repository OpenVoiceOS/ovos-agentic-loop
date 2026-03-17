# ovos-agentic-loop

Standalone `ChatEngine` plugins for OVOS that implement internal agent loops.
The key architectural insight: **SKILL.md and AGENTS.md are dual-purpose documents** — they serve Claude Code at dev-time and runtime LLM agents as tool descriptors and behavioral constraints.

## Components

| Class | Module | OPM Entry Point |
|---|---|---|
| `AgenticLoopEngine` | `ovos_agentic_loop.base` | `opm.agents.chat` |
| `ReActLoopEngine` | `ovos_agentic_loop.react` | `opm.agents.chat` |
| `SkillMDToolBox` | `ovos_agentic_loop.skills.toolbox` | `opm.agents.toolbox` |
| `SkillMDLoader` | `ovos_agentic_loop.skills.loader` | — |
| `AgentsMDContextManager` | `ovos_agentic_loop.context.agents_md` | `opm.agents.memory` |

## Architecture

```
SKILL.md frontmatter     →  AgentTool.name + AgentTool.description
SKILL.md body (markdown) →  sub-LLM system prompt for tool execution
AGENTS.md sections       →  AgentsMDContextManager system prompt
```

### AgenticLoopEngine (`base.py`)

Abstract `ChatEngine` subclass. Adds toolbox registration (`load_toolboxes`) and config-driven toolbox loading (`_load_toolboxes_from_config`).  All loop mechanics are encapsulated — callers see a plain `ChatEngine`.

### ReActLoopEngine (`react.py`)

Concrete ReAct (Reason + Act) implementation:
1. Prepend ReAct system prompt with tool schemas.
2. Brain LLM generates Thought + Action or `FINAL_ANSWER:`.
3. Execute tool → append Observation → repeat.
4. Stop on `FINAL_ANSWER:` or `max_iterations`.

Config: `brain` (ChatEngine plugin ID), `toolboxes` (list of ToolBox IDs), `max_iterations` (default 10).

### SkillMDToolBox (`skills/toolbox.py`)

`ToolBox` plugin exposing each installed SKILL.md as an `AgentTool`. Tool execution calls a sub-`ChatEngine` with the SKILL.md body as system prompt and the agent's `task` as the user message.

### SkillMDLoader (`skills/loader.py`)

Discovers SKILL.md files via:
1. `opm.agents.skill_md` entry-point group (explicit, zero-ambiguity).
2. Installed package data scan — walks all distributions for files named `SKILL.md`.

### AgentsMDContextManager (`context/agents_md.py`)

`AgentContextManager` that loads AGENTS.md files, filters sections by heading, and assembles the agent system prompt. Config: `agents_md_sources` (`["auto"]` or path list), `include_sections`, `system_prompt_prefix`.

## Persona config example

```json
{
  "name": "OVOSDeveloperAgent",
  "solvers": ["ovos-react-loop"],
  "memory_module": "ovos-agents-md-context-plugin",
  "plugin-config": {
    "ovos-react-loop": {
      "brain": "ovos-claude-plugin",
      "toolboxes": ["ovos-skill-md-toolbox"]
    },
    "ovos-agents-md-context-plugin": {
      "agents_md_sources": ["auto"],
      "include_sections": ["Universal Rules", "OpenVoiceOS Workspace"],
      "system_prompt_prefix": "You are an OVOS developer assistant."
    }
  }
}
```
