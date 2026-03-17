# QUICK_FACTS — ovos-agentic-loop

| Field | Value |
|---|---|
| **Package** | `ovos-agentic-loop` |
| **Version** | `0.1.0` |
| **Python** | ≥ 3.10 |
| **License** | Apache 2.0 |

## Key Classes

| Class | File | OPM Group |
|---|---|---|
| `AgenticLoopEngine` | `ovos_agentic_loop/base.py` | `opm.agents.chat` |
| `ReActLoopEngine` | `ovos_agentic_loop/react.py` | `opm.agents.chat` |
| `SkillMDToolBox` | `ovos_agentic_loop/skills/toolbox.py` | `opm.agents.toolbox` |
| `SkillMDLoader` | `ovos_agentic_loop/skills/loader.py` | — |
| `AgentsMDContextManager` | `ovos_agentic_loop/context/agents_md.py` | `opm.agents.memory` |
| `FileSystemToolBox` | `ovos_agentic_loop/tools/filesystem.py` | `opm.agents.toolbox` |
| `ShellToolBox` | `ovos_agentic_loop/tools/shell.py` | `opm.agents.toolbox` |
| `WebSearchToolBox` | `ovos_agentic_loop/tools/web.py` | `opm.agents.toolbox` |
| `ClockToolBox` | `ovos_agentic_loop/tools/clock.py` | `opm.agents.toolbox` |

## Entry Points

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

## Tests

73 unit tests in `test/`. Run: `uv run pytest test/ -v`
