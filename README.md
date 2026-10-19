# ovos-agentic-loop

[![PyPI](https://img.shields.io/pypi/v/ovos-agentic-loop)](https://pypi.org/project/ovos-agentic-loop/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)

This package lets an [OVOS](https://openvoiceos.org) persona use tools. It
wraps an LLM in an agent loop: the model asks for a tool, the loop runs it, and
the model answers with the tool's result. The tools can be OVOS tool plugins,
the tools of any MCP server, or the HTTP tools described by a UTCP manual.

It ships eight loop engines (native tool calling, ReAct, Plan-and-Execute,
Reflexion, Self-Ask, Chain-of-Thought, CRITIC and Tree-of-Thoughts), five
toolboxes, a SKILL.md toolbox, and an AGENTS.md context manager. Every part is
a standard [ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager)
plugin.

## What you can do

- Give a persona tools: a calculator, a clock, files, web search, or any OVOS
  tool plugin.
- Connect a persona to an MCP server or a UTCP manual through
  [ovos-tool-adapters](https://github.com/OpenVoiceOS/ovos-tool-adapters), for
  example the translation, speech-to-text and text-to-speech servers of OVOS.
- Use any OVOS chat plugin as the model behind the loop. No LLM is bundled.
- Turn installed `SKILL.md` files into tools, and build system prompts from
  `AGENTS.md` files.

## Installation

```bash
pip install ovos-agentic-loop

# Optional: web search support
pip install 'ovos-agentic-loop[web]'
```

The examples below also use
[ovos-openai-plugin](https://github.com/OpenVoiceOS/ovos-openai-plugin) as the
model, which works with any OpenAI-compatible endpoint (OpenAI, llama.cpp,
Ollama, vLLM and others), and ovos-tool-adapters for MCP and UTCP:

```bash
pip install ovos-openai-plugin 'ovos-tool-adapters[mcp,utcp]'
```

## User stories

### My persona should use the tools of an MCP server

You run an MCP server, for example
[ovos-translate-server](https://github.com/OpenVoiceOS/ovos-translate-server)
started with `--mcp`, and want your persona to call its tools.

Write the persona file. The loop is `ovos-native-toolcall-loop`, which uses
the model's own tool-calling API:

```json
{
  "name": "Translator",
  "solvers": ["ovos-native-toolcall-loop"],
  "ovos-native-toolcall-loop": {
    "brain": "ovos-chat-openai-plugin",
    "ovos-chat-openai-plugin": {
      "api_url": "http://127.0.0.1:4100/v1",
      "key": "sk-...",
      "model": "my-model"
    },
    "toolboxes": ["ovos-mcp-toolbox"],
    "ovos-mcp-toolbox": {"transport": "http", "url": "http://127.0.0.1:9686/mcp", "timeout": 30},
    "max_iterations": 5
  }
}
```

Load it with [ovos-persona](https://github.com/OpenVoiceOS/ovos-persona) and ask
a question:

```python
import json
from ovos_bus_client.session import Session
from ovos_persona import Persona
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

config = json.load(open("translator.json"))
persona = Persona(config["name"], config)
print(persona.chat([AgentMessage(role=MessageRole.USER,
                                 content="Translate 'the library opens at nine' from English into Portuguese with the translate tool.")],
                   Session("demo")))
```

The model called the server's `translate` tool, and the persona answered with
its result:

```text
tool call: ovos-mcp-toolbox.translate({"text": "the library opens at nine", "target_lang": "pt", "source_lang": "en"})
tool result: A biblioteca abre às nove.
answer: A biblioteca abre às nove.
```

`api_url` is the `/v1` root of the endpoint. The plugin adds
`/chat/completions` itself.

### My persona should use a UTCP manual

A UTCP manual describes plain HTTP endpoints as tools, so the persona calls the
server's normal API with no MCP layer. Every OVOS HTTP server publishes one at
`GET /utcp`.

```json
{
  "name": "Translator",
  "solvers": ["ovos-react-loop"],
  "ovos-react-loop": {
    "brain": "ovos-chat-openai-plugin",
    "ovos-chat-openai-plugin": {
      "api_url": "http://127.0.0.1:4100/v1",
      "key": "sk-...",
      "model": "my-model",
      "allow_system_prompts": true
    },
    "toolboxes": ["ovos-utcp-toolbox"],
    "ovos-utcp-toolbox": {
      "utcp_config": {
        "manual_call_templates": [
          {"name": "ovos_translate", "call_template_type": "http",
           "url": "http://127.0.0.1:9686/utcp", "http_method": "GET"}
        ]
      },
      "timeout": 30
    },
    "max_iterations": 5
  }
}
```

The same question then runs the manual's `translate_with_source` tool:

```text
tool call: ovos-utcp-toolbox.ovos_translate.translate_with_source({"src_lang": "en", "tgt_lang": "pt", "utterance": "the library opens at nine"})
tool result: A biblioteca abre às nove.
answer: A biblioteca abre às nove.
```

Two things differ from the MCP story:

- Use `ovos-react-loop` here, not `ovos-native-toolcall-loop`. A UTCP tool name
  carries the manual name and a dot (`ovos_translate.translate`), and the
  OpenAI tool-calling API accepts only letters, digits, `_` and `-` in a name.
  The provider refuses the request with `Function at index 0 has an invalid
  name`. The ReAct loop names tools in plain text, so the dot does no harm.
- Set `allow_system_prompts` to `true`. The ReAct family of loops puts the tool
  list in a system message, and ovos-openai-plugin drops system messages
  unless this key is set. Without it the model never sees the tools and
  answers from memory.

The UTCP toolbox needs an ovos-tool-adapters release that supports the UTCP 1.x
client (`utcp>=1.2.0`).

### I want a calculator and a clock from Python

```python
from ovos_agentic_loop.react import ReActLoopEngine
from ovos_agentic_loop.tools.math import MathToolBox
from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

engine = ReActLoopEngine({
    "brain": "ovos-chat-openai-plugin",
    "ovos-chat-openai-plugin": {
        "api_url": "http://localhost:11434/v1",
        "model": "my-model",
        "allow_system_prompts": True,
    },
    "max_iterations": 8,
})
engine.load_toolboxes([MathToolBox()])

response = engine.continue_chat([
    AgentMessage(role=MessageRole.USER, content="What is 17 * 23 + sqrt(144)?")
])
print(response.content)
```

Against an OpenAI-compatible endpoint, the loop ran
`evaluate_expression({"expression": "17 * 23 + sqrt(144)"})`, the tool returned
`403.0`, and the engine printed `403`.

To load toolboxes by name instead, list their ids in the engine config, for
example `"toolboxes": ["ovos-math-tools", "ovos-clock-tools"]`.

### A tool fails or its server goes away

The loop never stops on a tool error. It gives the error back to the model as
the tool's result, so the model can retry with other arguments or another
tool:

```text
Error: tool 'translate' failed: Tool execution failed for 'translate' in ToolBox 'ovos-mcp-toolbox': MCPError: Connection closed
```

After `max_iterations` tool rounds the loop makes one last call with no tools,
and the model answers from its own knowledge. In a live test with the MCP
server stopped in the middle of a conversation, the answer to "translate good
morning" was still `Bom dia`, with no sign to the user that the tool failed. Set `max_iterations` low if a slow
or dead server must not delay the answer.

When the MCP server is down while the persona loads, the toolbox loads no
tools and logs `MCPToolBox: failed to connect to MCP server`. The persona then
runs with no tools until it is loaded again.

## Loop architectures

| Entry point | Class | Best for |
|---|---|---|
| `ovos-native-toolcall-loop` | `NativeToolCallEngine` | Models with a tool-calling API. Falls back to the ReAct text loop for a model without one. |
| `ovos-react-loop` | `ReActLoopEngine` | General tool-using Q&A, with any model |
| `ovos-plan-execute-loop` | `PlanAndExecuteEngine` | Multi-step tasks requiring a plan |
| `ovos-reflexion-loop` | `ReflexionEngine` | Tasks requiring self-critique and retry |
| `ovos-self-ask-loop` | `SelfAskEngine` | Compositional questions needing sub-questions |
| `ovos-chain-of-thought-loop` | `ChainOfThoughtEngine` | Reasoning without tools (math, logic) |
| `ovos-critic-loop` | `CRITICEngine` | Factual tasks requiring claim verification |
| `ovos-tree-of-thoughts-loop` | `TreeOfThoughtsEngine` | Exploration-heavy problems (beam search) |

Every engine reads `brain` (the chat plugin id), a config section named after
the brain, `toolboxes` (a list of toolbox ids) and a config section per
toolbox. The ReAct, native tool-calling and Reflexion loops also read
`max_iterations` (default 10).

## Built-in toolboxes

| Entry point | Class | Tools |
|---|---|---|
| `ovos-math-tools` | `MathToolBox` | `evaluate_expression`, `unit_convert`, `statistics_summary`, `solve_equation` |
| `ovos-filesystem-tools` | `FileSystemToolBox` | `read_file`, `write_file`, `list_directory`, `search_in_files`, `find_files` |
| `ovos-shell-tools` | `ShellToolBox` | `run_command` (disabled by default; set `allow_shell: true` to enable) |
| `ovos-web-search-tools` | `WebSearchToolBox` | `web_search` (requires `ovos-agentic-loop[web]`) |
| `ovos-clock-tools` | `ClockToolBox` | `get_current_datetime` |
| `ovos-skill-md-toolbox` | `SkillMDToolBox` | One tool per installed SKILL.md |

These toolboxes are ordinary OVOS tool plugins. Any program on the device can
also run them over the messagebus through
[ovos-PHAL-plugin-tools](https://github.com/OpenVoiceOS/ovos-PHAL-plugin-tools).

## SKILL.md integration

Any package that ships a `SKILL.md` file is automatically discovered and exposed as an agent tool:

```markdown
---
name: my-skill
description: Does something useful.
---
You are a helpful assistant specialised in...
```

The tool name is the slugified `name` field. The SKILL.md body becomes the system prompt for a sub-LLM call.

## AGENTS.md context management

`AgentsMDContextManager` assembles system prompts from `AGENTS.md` files at runtime. These are the same files Claude Code reads at dev-time:

```python
from ovos_agentic_loop.context.agents_md import AgentsMDContextManager

ctx = AgentsMDContextManager({
    "agents_md_sources": ["auto"],          # auto-discover from installed packages
    "include_sections": ["Rules", "Style"], # filter to specific headings
})
messages = ctx.build_conversation_context(utterance, session_id="s1")
```

## Security notes

- **`ShellToolBox`**: `allow_shell` defaults to `False`. Only enable with fully-trusted LLMs. The command string is passed directly to `/bin/sh`.
- **`FileSystemToolBox`**: every path must resolve inside `root_path`, and the default `root_path` is the working directory of the process. `allow_write` defaults to `True`. Set `root_path` to a directory made for the agent, and set `allow_write` to `False` for an agent that only reads.
- **`MathToolBox`**: uses `ast.parse` with an allowlist. `eval()` is never called.
- A loaded toolbox is also open to every messagebus client when ovos-PHAL-plugin-tools runs on the same device.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| The persona answers `Error: no brain configured.` | The `brain` key is missing, or the chat plugin it names is not installed. Check the log for `failed to load brain`. |
| `404 Client Error ... /chat/completions/chat/completions` | `api_url` ends in `/chat/completions`. Give the `/v1` root only. |
| The log shows `dropped 1 system message(s)` and the model never calls a tool | Set `allow_system_prompts: true` in the brain's config section. The ReAct family of loops needs the system message. |
| The provider answers `Function at index 0 has an invalid name` | A tool name holds a character outside `a-z A-Z 0-9 _ -`, as UTCP names do. Use `ovos-react-loop` for that toolbox. |
| `MCPToolBox: failed to connect to MCP server` | The MCP server was not reachable when the persona loaded. Start the server first, then load the persona. |

## Documentation

- [docs/loop-architectures.md](docs/loop-architectures.md): all loop engines with algorithm details
- [docs/native-toolcall-loop.md](docs/native-toolcall-loop.md): the native tool-calling loop
- [docs/react-loop.md](docs/react-loop.md): ReAct prompt format and parsing
- [docs/toolboxes.md](docs/toolboxes.md): all toolbox schemas and config keys
- [docs/skill-md.md](docs/skill-md.md): SKILL.md authoring and packaging guide
- [docs/agents-md.md](docs/agents-md.md): AGENTS.md context manager reference
- [docs/opm-integration.md](docs/opm-integration.md): OPM entry points and persona wiring

## License

Apache 2.0. See [LICENSE](LICENSE).

## Credits

Developed by [TigreGotico](https://tigregotico.pt) for [OpenVoiceOS](https://openvoiceos.org).

Funded by [NGI0 Commons Fund](https://nlnet.nl/project/OpenVoiceOS) / [NLnet](https://nlnet.nl)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429),
through the European Commission's [Next Generation Internet](https://ngi.eu) programme.
