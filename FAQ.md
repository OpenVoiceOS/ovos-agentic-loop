# FAQ — ovos-agentic-loop

**Q: What is ovos-agentic-loop?**
A: A `ChatEngine` plugin package implementing internal agent loops for OVOS. It provides `AgenticLoopEngine` (abstract base), `ReActLoopEngine` (ReAct pattern), `SkillMDToolBox` (SKILL.md → AgentTool), and `AgentsMDContextManager` (AGENTS.md → system prompt).

**Q: How does it relate to ovos-plugin-manager?**
A: It registers as `opm.agents.chat`, `opm.agents.toolbox`, and `opm.agents.memory` plugins. A persona service sees `ReActLoopEngine` as a plain `ChatEngine` — all loop mechanics are internal.

**Q: What is the difference between AgenticLoopEngine and ChatEngine?**
A: `AgenticLoopEngine` inherits `ChatEngine` and adds toolbox registration (`load_toolboxes`, `_load_toolboxes_from_config`). The interface to callers is identical.

**Q: How does ReActLoopEngine work?**
A: Each iteration: (1) build system prompt with available tool schemas, (2) call brain LLM, (3) parse Action or `FINAL_ANSWER:`, (4) execute tool and append Observation, repeat until final answer or `max_iterations`.

**Q: What is SkillMDToolBox?**
A: A `ToolBox` plugin that discovers installed SKILL.md files and exposes each as an `AgentTool`. Invoking a tool calls a sub-ChatEngine (brain) with the SKILL.md body as system prompt.

**Q: How are SKILL.md files discovered?**
A: Two strategies in order: (1) `opm.agents.skill_md` entry-point group — packages declare their SKILL.md path explicitly; (2) package-data scan — walks all installed distributions for files named `SKILL.md`.

**Q: How do I declare a SKILL.md entry point?**
A: Add to `pyproject.toml`:
```toml
[project.entry-points."opm.agents.skill_md"]
my-skill = "my_package:SKILL_MD_PATH"
```
where `SKILL_MD_PATH` is an importable string attribute holding the file path.

**Q: What must a SKILL.md file contain?**
A: YAML frontmatter with at least `name` and `description` fields, followed by a markdown body used as the sub-LLM system prompt:
```
---
name: my-skill
description: One-line description shown to the agent as the tool description.
---
## Instructions
...
```

**Q: What is AgentsMDContextManager?**
A: An `AgentContextManager` that loads AGENTS.md files, filters to selected sections, and assembles the LLM system prompt. The same rules that govern Claude Code govern the runtime agent.

**Q: How do I limit which AGENTS.md sections are used?**
A: Set `include_sections` in the plugin config:
```json
{"include_sections": ["Universal Rules", "OpenVoiceOS Workspace"]}
```
Matching is case-insensitive substring. Empty list = include all sections.

**Q: How do I configure the brain LLM for SkillMDToolBox?**
A: Call `toolbox.set_brain(brain_instance)` or let the owning `ReActLoopEngine` inject it. The toolbox itself does not load a brain from config.

**Q: What does max_iterations do in ReActLoopEngine?**
A: Limits tool-call cycles. When reached, the brain is asked for a final answer with a prompt. Default: 10.
