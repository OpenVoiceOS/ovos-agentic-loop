# ovos-agentic-loop — Examples

## weatherman_persona.py

A minimal demo wiring `ReActLoopEngine` + `WeatherToolBox` + a local LLM.

### What it demonstrates

- How to instantiate `ReActLoopEngine` directly in Python.
- How to inject a `ChatEngine` brain (OpenAI-compatible local LLM).
- How to attach a `ToolBox` (`WeatherToolBox` from `ovos-skill-weather`).
- A simple REPL loop that preserves conversation history.
- The equivalent JSON persona config for use with `ovos-persona`.

### Architecture

```
User input
    │
    ▼
ReActLoopEngine
    │  injects location context
    │  sends messages to brain LLM
    │
    ├──► Thought (LLM decides to call a tool)
    │       │
    │       ▼
    │   WeatherToolBox.call_tool("get_current_weather", {...})
    │       │
    │       ▼
    │   Open-Meteo API (free, no key)
    │       │
    │       ▼
    │   Observation appended to message history
    │
    └──► FINAL_ANSWER (LLM composes weather reply)
             │
             ▼
         User sees the answer
```

### Prerequisites

```bash
pip install ovos-agentic-loop ovos-openai-plugin ovos-skill-weather
```

Start a local OpenAI-compatible LLM (any of these work):

```bash
# Ollama
ollama serve
ollama pull llama3

# OR llama.cpp server
# OR LM Studio
# OR point at the real OpenAI API
```

### Run

```bash
python examples/weatherman_persona.py
```

Print the equivalent persona JSON config:

```bash
python examples/weatherman_persona.py --print-config
```

### Environment variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `WEATHERMAN_MODEL` | `llama3` | Model name to request from the LLM server |
| `WEATHERMAN_API_URL` | `http://localhost:11434/v1` | Base URL of the OpenAI-compatible server |
| `WEATHERMAN_LAT` | `48.8566` | Default latitude (Paris) |
| `WEATHERMAN_LON` | `2.3522` | Default longitude |
| `WEATHERMAN_TZ` | `Europe/Paris` | Default timezone |
| `WEATHERMAN_UNITS` | `metric` | `metric` or `imperial` |

### Using with ovos-persona

Load `weatherman_persona.json` into your OVOS persona directory, or use the
`PERSONA_CONFIG` dict from `weatherman_persona.py` directly:

```python
from examples.weatherman_persona import PERSONA_CONFIG
from ovos_plugin_manager.persona import load_persona

persona = load_persona("WeatherMan", PERSONA_CONFIG)
response = persona.chat("Will it rain tomorrow in London?")
```
