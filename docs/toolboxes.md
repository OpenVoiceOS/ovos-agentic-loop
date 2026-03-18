# ToolBox Plugin Reference

All `ToolBox` classes implement the abstract `ToolBox` base from OPM (`agent_tools.py:56`). Each registers under `opm.agents.toolbox` and implements the single required method `discover_tools() -> List[AgentTool]`.

---

## SkillMDToolBox

**Entry point ID**: `ovos-skill-md-toolbox`
**Source**: `ovos_agentic_loop/skills/toolbox.py:48`

Converts installed SKILL.md files into `AgentTool` instances. Each SKILL.md yields one tool; tool execution calls a sub-`ChatEngine` with the SKILL.md body as the system prompt.

### Tool: (one per SKILL.md, dynamically generated)

Tool name is slugified from the SKILL.md `name` frontmatter field via `_slugify()` — `toolbox.py:12`:
- Lower-case, non-alphanumeric characters replaced with underscores, leading/trailing underscores stripped.
- Example: `"web-search"` → `"web_search"`.

| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `task` | str | yes | Free-text instruction for the skill |
| `context` | str | no | Additional context or constraints |

| Output field | Type | Description |
| :--- | :--- | :--- |
| `result` | str | Natural-language response from the sub-LLM |
| `skill_used` | str | The SKILL.md `name` field of the invoked skill |

### Tool Execution (`_invoke_skill`) — `toolbox.py:96`

1. Raises `RuntimeError` if no brain has been set.
2. Builds user content: `args.task` if no context, else `"{task}\n\nContext: {context}"`.
3. Assembles `[SYSTEM: <skill body>, USER: <user_content>]` and calls `self._brain.continue_chat(messages)`.
4. Returns `SkillCallOutput(result=response.content, skill_used=entry.name)`.

### Config Keys

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `extra_skill_md_paths` | List[str] | `[]` | Additional SKILL.md paths to load |

### Brain Injection

`set_brain(brain: ChatEngine)` — `toolbox.py:87`. Must be called before any tool invocation. The owning `ReActLoopEngine` does **not** auto-share its brain; manual wiring is required (see AUDIT ISSUE-002).

### Example Persona Config

```json
{
  "ovos-skill-md-toolbox": {
    "extra_skill_md_paths": ["/home/user/custom/SKILL.md"]
  }
}
```

---

## FileSystemToolBox

**Entry point ID**: `ovos-filesystem-tools`
**Source**: `ovos_agentic_loop/tools/filesystem.py:85`

Provides five tools for local filesystem interaction.

### Tool: `read_file`

| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `path` | str | yes | Absolute or relative path to read |

| Output field | Type | Description |
| :--- | :--- | :--- |
| `content` | str | UTF-8 file contents or error message |
| `path` | str | Resolved absolute path |

On any `Exception`, returns the exception message in `content` rather than raising — `filesystem.py:127`.

### Tool: `write_file`

Guarded by `allow_write` config flag — `filesystem.py:140`.

| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `path` | str | yes | Target file path (parent dirs created automatically) |
| `content` | str | yes | UTF-8 text to write |

| Output field | Type | Description |
| :--- | :--- | :--- |
| `success` | bool | `True` if write succeeded |
| `path` | str | Resolved absolute path |
| `message` | str | Human-readable status |

When `allow_write=False`, the tool is still registered in `discover_tools()` — `filesystem.py:264` — with a description noting it is disabled, so the LLM knows the tool exists but is unavailable. The `_write_file` handler checks the flag at call time and returns `success=False`.

### Tool: `list_directory`

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | str | — | Directory to list |
| `pattern` | str | `"*"` | Shell glob filter |

| Output field | Type | Description |
| :--- | :--- | :--- |
| `entries` | List[str] | Matching filesystem entry paths (sorted) |
| `path` | str | Resolved directory path |

### Tool: `search_in_files`

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `pattern` | str | — | Regular expression to search |
| `path` | str | `"."` | Root directory |
| `glob` | str | `"**/*"` | Glob to select files |

| Output field | Type | Description |
| :--- | :--- | :--- |
| `matches` | List[Dict] | Each dict: `{file, line_number, line}` |
| `total` | int | Count of matching lines |

Invalid regex returns empty results without raising — `filesystem.py:186`.

### Tool: `find_files`

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `glob` | str | — | Glob pattern (e.g. `**/*.py`) |
| `path` | str | `"."` | Root directory |

| Output field | Type | Description |
| :--- | :--- | :--- |
| `files` | List[str] | Matched file paths (files only, sorted) |
| `total` | int | Count of matched files |

### Config Keys

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `allow_write` | bool | `True` | Set to `False` for read-only agents |

### Example Persona Config

```json
{
  "ovos-filesystem-tools": {
    "allow_write": false
  }
}
```

---

## ShellToolBox

**Entry point ID**: `ovos-shell-tools`
**Source**: `ovos_agentic_loop/tools/shell.py:26`

Exposes one tool for executing shell commands.

### Tool: `run_command`

Uses `subprocess.run(..., shell=True, ...)` — `shell.py:75`.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `command` | str | — | Shell command to execute |
| `cwd` | str | `"."` | Working directory |
| `timeout` | int | `30` | Requested timeout (seconds); capped at `max_timeout` config |

| Output field | Type | Description |
| :--- | :--- | :--- |
| `stdout` | str | Standard output |
| `stderr` | str | Standard error (also used for error messages when blocked/timed-out) |
| `returncode` | int | Exit code (`-1` on timeout or error) |
| `success` | bool | `True` if `returncode == 0` |

Timeout cap: `effective_timeout = min(args.timeout, config["max_timeout"])` — `shell.py:72`.

On `TimeoutExpired`, returns `returncode=-1` and a message in `stderr` — `shell.py:90`.

When `allow_shell=False`, returns immediately with `returncode=-1` and `"Shell execution is disabled"` in `stderr` — `shell.py:63`.

### Config Keys

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `allow_shell` | bool | `False` | Must be explicitly set to `True` to enable execution |
| `max_timeout` | int | `120` | Maximum allowed timeout in seconds |
| `allowed_commands` | list[str] | `[]` | Permitted command prefixes. Empty = all allowed. Non-empty = only matching prefixes execute — `shell.py:85` |

When `allowed_commands` is non-empty, the first word of the command is matched against each prefix. Non-matching commands return `returncode=-1` and `"not permitted"` in `stderr` without any subprocess execution.

### Security Note

`shell=True` passes the command string directly to `/bin/sh`. Always combine `allow_shell: true` with a tight `allowed_commands` list in production environments.

### Example Persona Config

```json
{
  "ovos-shell-tools": {
    "allow_shell": true,
    "allowed_commands": ["git", "ls", "cat"],
    "max_timeout": 30
  }
}
```

---

## WebSearchToolBox

**Entry point ID**: `ovos-web-search-tools`
**Source**: `ovos_agentic_loop/tools/web.py:25`

DuckDuckGo text search. Requires optional dependency `duckduckgo-search>=6.0`.

Install: `pip install 'ovos-agentic-loop[web]'`

### Tool: `web_search`

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `query` | str | — | Search query string |
| `max_results` | int | `5` | Maximum results to return |

| Output field | Type | Description |
| :--- | :--- | :--- |
| `results` | List[Dict] | Each dict: `{title, url, snippet}` |
| `query` | str | The query that was executed |

If `duckduckgo-search` is not installed, returns one result with `title="Package not installed"` and an install instruction in `snippet` — `web.py:65`. No exception is raised; the LLM receives a friendly error observation.

If the search call itself fails, returns one result with `title="Search error"` and the exception message in `snippet` — `web.py:93`.

URL field normalisation: tries `r.get("href", r.get("url", ""))` to handle API key differences across package versions — `web.py:84`.

### Config Keys

None. The config dict is accepted but unused.

### Example Persona Config

```json
{
  "ovos-web-search-tools": {}
}
```

---

## ClockToolBox

**Entry point ID**: `ovos-clock-tools`
**Source**: `ovos_agentic_loop/tools/clock.py:22`

No external dependencies. Returns the current system date and time.

### Tool: `get_current_datetime`

No input arguments.

| Output field | Type | Description |
| :--- | :--- | :--- |
| `iso` | str | Full ISO-8601 datetime with timezone offset |
| `date` | str | `YYYY-MM-DD` |
| `time` | str | `HH:MM:SS` |
| `timezone` | str | System timezone name (e.g. `"UTC"`, `"EST"`) or UTC offset string |

Implementation: `datetime.now().astimezone()` — `clock.py:53`. Timezone name falls back to `str(now.utcoffset())` if `tzname()` returns empty — `clock.py:54`.

### Config Keys

None.


---

## MathToolBox

**Entry point ID**: `ovos-math-tools`
**Source**: `ovos_agentic_loop/tools/math.py`

Safe mathematical operations. No external dependencies required for three of the four tools; `solve_equation` uses `sympy` when installed.

### Tool: `evaluate_expression`

| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `expression` | str | yes | Math expression string |

| Output field | Type | Description |
| :--- | :--- | :--- |
| `result` | float | Numeric result (`nan` on error) |
| `expression` | str | Original input |
| `error` | str\|None | Error message on failure |

Evaluation uses `ast.parse` + a whitelist of operators and functions — no `eval()` — `math.py:71`. Supported: `+`, `-`, `*`, `/`, `//`, `%`, `**`, and functions `abs, round, sqrt, ceil, floor, log, log10, log2, exp, sin, cos, tan, asin, acos, atan, atan2, degrees, radians, factorial, gcd, lcm`. Constants: `pi, e, tau, inf`.

### Tool: `unit_convert`

| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `value` | float | yes | Numeric value to convert |
| `from_unit` | str | yes | Source unit (case-insensitive) |
| `to_unit` | str | yes | Target unit (case-insensitive) |

| Output field | Type | Description |
| :--- | :--- | :--- |
| `result` | float | Converted value |
| `category` | str | Measurement category |
| `error` | str\|None | Error on failure |

Supported categories: `length`, `mass`, `volume`, `time`, `speed`, `area`, `data`, `temperature`. Temperature uses affine conversion (Celsius/Fahrenheit/Kelvin); all others use SI-base linear factors — `math.py:136`. Incompatible categories return an error.

### Tool: `statistics_summary`

| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `numbers` | list[float] | yes | Values to summarise |

| Output field | Type | Description |
| :--- | :--- | :--- |
| `count` | int | Number of values |
| `mean` | float | Arithmetic mean |
| `median` | float | Median |
| `stdev` | float\|None | Sample std deviation (None when n < 2) |
| `minimum` | float | Min value |
| `maximum` | float | Max value |
| `total` | float | Sum |

### Tool: `solve_equation`

| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `equation` | str | yes | Python expression equal to zero (e.g. `x**2 - 4`) |

| Output field | Type | Description |
| :--- | :--- | :--- |
| `solutions` | list[float] | Real-valued roots |
| `method` | str | `"symbolic"` (sympy) or `"numeric"` (bisection fallback) |
| `error` | str\|None | Error on failure |

Tries `sympy.solve` first; falls back to bisection scan over `[-1000, 1000]` — `math.py:375`.

### Config Keys

None.
