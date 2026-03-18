# ovos-agentic-loop — Examples

| Example | Engine | Tools | Use case |
| :--- | :--- | :--- | :--- |
| `weatherman_persona.py` | `ReActLoopEngine` | `WeatherToolBox` | Real-time weather queries |
| `research_assistant_persona.py` | `ReActLoopEngine` | `WebSearchToolBox`, `ClockToolBox` | Current events, fact lookup |
| `chain_of_thought_persona.py` | `ChainOfThoughtEngine` | none | Reasoning, math, logic |
| `reflexion_persona.py` | `ReflexionEngine` | `WebSearchToolBox` | Self-correcting, high-accuracy answers |

Each example ships a matching `.json` persona config for use with `ovos-persona`.

---

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

### Environment variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `WEATHERMAN_MODEL` | `llama3` | Model name |
| `WEATHERMAN_API_URL` | `http://localhost:11434/v1` | LLM server URL |
| `WEATHERMAN_LAT` | `48.8566` | Default latitude (Paris) |
| `WEATHERMAN_LON` | `2.3522` | Default longitude |
| `WEATHERMAN_TZ` | `Europe/Paris` | Default timezone |
| `WEATHERMAN_UNITS` | `metric` | `metric` or `imperial` |

---

## research_assistant_persona.py

`ReActLoopEngine` + `WebSearchToolBox` (DuckDuckGo) + `ClockToolBox`.

Best for questions that require current information: news, recent events,
live data, or "what time is it".

### Architecture

```
User input
    │
    ▼
ReActLoopEngine (max 6 iterations)
    ├──► web_search("query") → DuckDuckGo results
    ├──► get_current_datetime() → system clock
    └──► FINAL_ANSWER
```

### Prerequisites

```bash
pip install "ovos-agentic-loop[web]" ovos-openai-plugin
```

### Environment variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `RESEARCH_MODEL` | `llama3` | Model name |
| `RESEARCH_API_URL` | `http://localhost:11434/v1` | LLM server URL |
| `RESEARCH_RESULTS` | `5` | Max search results per query |

---

## chain_of_thought_persona.py

`ChainOfThoughtEngine` + a local LLM.  No tools — single LLM call.

Best for arithmetic, logic puzzles, multi-step instructions, and any task
where structured reasoning (not external data) is the bottleneck.

### Architecture

```
User input
    │
    ▼
ChainOfThoughtEngine
    │  "Let's think step by step…"
    │  single LLM call
    └──► FINAL ANSWER
```

### Prerequisites

```bash
pip install ovos-agentic-loop ovos-openai-plugin
```

### Environment variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `COT_MODEL` | `llama3` | Model name |
| `COT_API_URL` | `http://localhost:11434/v1` | LLM server URL |

---

## reflexion_persona.py

`ReflexionEngine` + `WebSearchToolBox`.

After each episode the engine evaluates its own answer.  If the answer is
weak, it generates a verbal critique (reflection) and retries — up to
`max_reflections` times.  Each retry is conditioned on accumulated lessons
learned.

Best for multi-hop questions, tasks where the first attempt is likely
wrong, or any situation where accuracy matters more than latency.

### Architecture

```
User input
    │
    ▼
ReflexionEngine
    │
    ├── Episode 1: inner ReAct loop → draft answer
    │       │
    │       ▼
    │   Evaluator: "Is this answer satisfactory?"
    │       │ No → Reflector: "What went wrong?"
    │       │             │
    │       │         reflection stored
    │       │
    ├── Episode 2: ReAct loop (with reflection in context) → revised answer
    │       ▼
    │   Evaluator: "Satisfactory?" → Yes → done
    │
    └──► Best answer returned
```

### Prerequisites

```bash
pip install "ovos-agentic-loop[web]" ovos-openai-plugin
```

### Environment variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `REFLEXION_MODEL` | `llama3` | Model name |
| `REFLEXION_API_URL` | `http://localhost:11434/v1` | LLM server URL |
| `REFLEXION_MAX_ROUNDS` | `3` | Max self-reflection episodes |

---

## Common setup

Start a local OpenAI-compatible LLM (any of these work):

```bash
# Ollama
ollama serve
ollama pull llama3

# OR llama.cpp server
# OR LM Studio
# OR point at the real OpenAI API
```

Run any example:

```bash
python examples/<example>.py

# Print the equivalent JSON persona config and exit:
python examples/<example>.py --print-config
```

## Using with ovos-persona

Load any `.json` config from this directory into your OVOS persona directory,
or use the `PERSONA_CONFIG` dict from the corresponding `.py` file directly:

```python
from ovos_plugin_manager.persona import load_persona
from examples.research_assistant_persona import PERSONA_CONFIG

persona = load_persona("Researcher", PERSONA_CONFIG)
response = persona.chat("What happened in the news today?")
```
