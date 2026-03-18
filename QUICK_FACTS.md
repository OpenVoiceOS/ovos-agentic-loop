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
| `AgenticLoopEngine` | `ovos_agentic_loop/base.py` | `opm.agents.chat` (abstract base) |
| `ReActLoopEngine` | `ovos_agentic_loop/react.py` | `opm.agents.chat` |
| `PlanAndExecuteEngine` | `ovos_agentic_loop/plan_execute.py` | `opm.agents.chat` |
| `ReflexionEngine` | `ovos_agentic_loop/reflexion.py` | `opm.agents.chat` |
| `SelfAskEngine` | `ovos_agentic_loop/self_ask.py` | `opm.agents.chat` |
| `ChainOfThoughtEngine` | `ovos_agentic_loop/chain_of_thought.py` | `opm.agents.chat` |
| `CRITICEngine` | `ovos_agentic_loop/critic.py` | `opm.agents.chat` |
| `TreeOfThoughtsEngine` | `ovos_agentic_loop/tree_of_thoughts.py` | `opm.agents.chat` |
| `SkillMDToolBox` | `ovos_agentic_loop/skills/toolbox.py` | `opm.agents.toolbox` |
| `SkillMDLoader` | `ovos_agentic_loop/skills/loader.py` | — |
| `AgentsMDContextManager` | `ovos_agentic_loop/context/agents_md.py` | `opm.agents.memory` |
| `FileSystemToolBox` | `ovos_agentic_loop/tools/filesystem.py` | `opm.agents.toolbox` |
| `ShellToolBox` | `ovos_agentic_loop/tools/shell.py` | `opm.agents.toolbox` |
| `WebSearchToolBox` | `ovos_agentic_loop/tools/web.py` | `opm.agents.toolbox` |
| `ClockToolBox` | `ovos_agentic_loop/tools/clock.py` | `opm.agents.toolbox` |
| `MathToolBox` | `ovos_agentic_loop/tools/math.py` | `opm.agents.toolbox` |

## Entry Points

```toml
[project.entry-points."opm.agents.chat"]
ovos-react-loop            = "ovos_agentic_loop.factory:ReActLoopEnginePlugin"
ovos-plan-execute-loop     = "ovos_agentic_loop.factory:PlanAndExecuteEnginePlugin"
ovos-reflexion-loop        = "ovos_agentic_loop.factory:ReflexionEnginePlugin"
ovos-self-ask-loop         = "ovos_agentic_loop.factory:SelfAskEnginePlugin"
ovos-chain-of-thought-loop = "ovos_agentic_loop.factory:ChainOfThoughtEnginePlugin"
ovos-critic-loop           = "ovos_agentic_loop.factory:CRITICEnginePlugin"
ovos-tree-of-thoughts-loop = "ovos_agentic_loop.factory:TreeOfThoughtsEnginePlugin"

[project.entry-points."opm.agents.toolbox"]
ovos-skill-md-toolbox   = "ovos_agentic_loop.skills.toolbox:SkillMDToolBox"
ovos-filesystem-tools   = "ovos_agentic_loop.tools.filesystem:FileSystemToolBox"
ovos-shell-tools        = "ovos_agentic_loop.tools.shell:ShellToolBox"
ovos-web-search-tools   = "ovos_agentic_loop.tools.web:WebSearchToolBox"
ovos-clock-tools        = "ovos_agentic_loop.tools.clock:ClockToolBox"
ovos-math-tools         = "ovos_agentic_loop.tools.math:MathToolBox"

[project.entry-points."opm.agents.memory"]
ovos-agents-md-context-plugin = "ovos_agentic_loop.context.agents_md:AgentsMDContextManager"
```

## Tests

276 unit tests in `test/`. Run: `uv run pytest test/ -v`
