# ovos-agentic-loop

[![PyPI](https://img.shields.io/pypi/v/ovos-agentic-loop)](https://pypi.org/project/ovos-agentic-loop/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)

Agent-loop `ChatEngine` plugins for [OVOS](https://openvoiceos.org). Implements seven agentic reasoning patterns (ReAct, Plan-and-Execute, Reflexion, Self-Ask, Chain-of-Thought, CRITIC, Tree-of-Thoughts), five built-in toolboxes, SKILL.md integration, and AGENTS.md context management — all as standard OPM plugins.

---

## Features

- **7 loop architectures** — from simple Chain-of-Thought to full Reflexion and Tree-of-Thoughts beam search
- **5 built-in toolboxes** — filesystem, shell, web search, clock, math (calculator + unit conversion + statistics)
- **SKILL.md toolbox** — turns any installed SKILL.md file into an agent tool
- **AGENTS.md context manager** — loads structured system prompts from AGENTS.md files at runtime
- **No LLM bundled** — wire any OPM `ChatEngine` as the inner brain
- **No persona bundled** — compose your own persona using the provided primitives

---

## Installation

```bash
pip install ovos-agentic-loop

# Optional: web search support
pip install 'ovos-agentic-loop[web]'
```

---

## Quick Start

### Minimal ReAct agent with a calculator

```python
from ovos_agentic_loop.react import ReActLoopEngine
from ovos_agentic_loop.tools.math import MathToolBox
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

# Wire a local LLM as the brain (any OPM ChatEngine works)
engine = ReActLoopEngine({
    "brain": "ovos-chat-openai-plugin",
    "ovos-chat-openai-plugin": {
        "api_url": "http://localhost:11434/v1/chat/completions",
    },
    "max_iterations": 8,
})
engine.load_toolboxes([MathToolBox()])

response = engine.continue_chat([
    AgentMessage(role=MessageRole.USER, content="What is 17 * 23 + sqrt(144)?")
])
print(response.content)
```

### Persona JSON (all-in-one config)

```json
{
  "name": "MyAgent",
  "solvers": ["ovos-react-loop"],
  "ovos-react-loop": {
    "brain": "ovos-chat-openai-plugin",
    "ovos-chat-openai-plugin": {
      "api_url": "http://localhost:11434/v1/chat/completions"
    },
    "toolboxes": ["ovos-math-tools", "ovos-web-search-tools", "ovos-clock-tools"],
    "max_iterations": 10
  }
}
```

---

## Loop Architectures

| Entry point | Class | Best for |
|---|---|---|
| `ovos-react-loop` | `ReActLoopEngine` | General tool-using Q&A |
| `ovos-plan-execute-loop` | `PlanAndExecuteEngine` | Multi-step tasks requiring a plan |
| `ovos-reflexion-loop` | `ReflexionEngine` | Tasks requiring self-critique and retry |
| `ovos-self-ask-loop` | `SelfAskEngine` | Compositional questions needing sub-questions |
| `ovos-chain-of-thought-loop` | `ChainOfThoughtEngine` | Reasoning without tools (math, logic) |
| `ovos-critic-loop` | `CRITICEngine` | Factual tasks requiring claim verification |
| `ovos-tree-of-thoughts-loop` | `TreeOfThoughtsEngine` | Exploration-heavy problems (beam search) |

---

## Built-in Toolboxes

| Entry point | Class | Tools |
|---|---|---|
| `ovos-math-tools` | `MathToolBox` | `evaluate_expression`, `unit_convert`, `statistics_summary`, `solve_equation` |
| `ovos-filesystem-tools` | `FileSystemToolBox` | `read_file`, `write_file`, `list_directory`, `search_in_files`, `find_files` |
| `ovos-shell-tools` | `ShellToolBox` | `run_command` (disabled by default; set `allow_shell: true` to enable) |
| `ovos-web-search-tools` | `WebSearchToolBox` | `web_search` (requires `ovos-agentic-loop[web]`) |
| `ovos-clock-tools` | `ClockToolBox` | `get_current_datetime` |
| `ovos-skill-md-toolbox` | `SkillMDToolBox` | One tool per installed SKILL.md |

---

## SKILL.md Integration

Any package that ships a `SKILL.md` file is automatically discovered and exposed as an agent tool:

```markdown
---
name: my-skill
description: Does something useful.
---
You are a helpful assistant specialised in...
```

The tool name is the slugified `name` field; the SKILL.md body becomes the system prompt for a sub-LLM call.

---

## AGENTS.md Context Management

`AgentsMDContextManager` assembles system prompts from `AGENTS.md` files at runtime — the same files Claude Code reads at dev-time:

```python
from ovos_agentic_loop.context.agents_md import AgentsMDContextManager

ctx = AgentsMDContextManager({
    "agents_md_sources": ["auto"],          # auto-discover from installed packages
    "include_sections": ["Rules", "Style"], # filter to specific headings
})
messages = ctx.build_conversation_context(utterance, session_id="s1")
```

---

## Security Notes

- **`ShellToolBox`** — `allow_shell` defaults to `False`. Only enable with fully-trusted LLMs; the command string is passed directly to `/bin/sh`.
- **`FileSystemToolBox`** — set `root_path` to restrict file access to a subtree. Without it, the agent can read any world-readable file.
- **`MathToolBox`** — uses `ast.parse` with an allowlist; `eval()` is never called.

---

## Documentation

- [docs/loop-architectures.md](docs/loop-architectures.md) — all 7 loop engines with algorithm details
- [docs/react-loop.md](docs/react-loop.md) — ReAct prompt format and parsing
- [docs/toolboxes.md](docs/toolboxes.md) — all toolbox schemas and config keys
- [docs/skill-md.md](docs/skill-md.md) — SKILL.md authoring and packaging guide
- [docs/agents-md.md](docs/agents-md.md) — AGENTS.md context manager reference
- [docs/opm-integration.md](docs/opm-integration.md) — OPM entry points and persona wiring

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
