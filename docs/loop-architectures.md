# Agent Loop Architectures

`ovos-agentic-loop` ships four distinct loop strategies. Each is a concrete
`AgenticLoopEngine` (`ovos_agentic_loop/base.py:8`) and is registered as an
`opm.agents.chat` entry point so `ovos-persona` can load it by ID.

---

## Choosing a loop

| Use-case signal | Recommended loop |
| :--- | :--- |
| Single-turn tool use, general assistant | **ReAct** |
| Multi-step task with distinct, parallelisable sub-goals | **Plan-and-Execute** |
| Correctness matters; agent may fail on first attempt | **Reflexion** |
| Multi-hop knowledge question (chain of facts) | **Self-Ask** |

The loops are not mutually exclusive.  Reflexion *wraps* ReAct internally, so
it inherits every ReAct capability while adding the self-correction outer loop.
Plan-and-Execute uses its own mini-ReAct sub-loop per step.

---

## ReAct — Reason + Act

**Entry point:** `ovos-react-loop`
**Class:** `ReActLoopEngine` — `ovos_agentic_loop/react.py:92`
**Paper:** Yao et al., 2022 — *ReAct: Synergizing Reasoning and Acting in Language Models*

### How it works

Every iteration the LLM produces a **Thought → Action → Observation** triplet:

```
Thought: I need the current temperature in Paris.
Action: get_current_weather
Action Input: {"latitude": 48.85, "longitude": 2.35, "timezone": "Europe/Paris"}
Observation: {"temperature": 18, "condition_description": "Partly cloudy", ...}

Thought: I have the data. I can answer now.
FINAL_ANSWER: It is currently 18 °C and partly cloudy in Paris.
```

The loop exits on `FINAL_ANSWER:` or when `max_iterations` is exhausted
(at which point the LLM is asked for its best answer).

### Loop logic (`react.py:204–250`)

```
loop_messages = [react_system_prompt] + conversation_history

for _ in range(max_iterations):
    response = brain.continue_chat(loop_messages)
    if FINAL_ANSWER in response:  → return answer
    if Action found:
        result = call_tool(action)
        append (assistant: response) + (user: "Observation: {result}")
    else:
        return response as-is

# fallback: ask brain for FINAL_ANSWER
```

### Strengths and limits

**Strengths:** Simple, predictable, single-LLM-call per iteration.
**Limits:** The LLM must commit to a complete plan on each turn; it cannot
backtrack if an early tool call was wrong.  Repeated tool-call failures
consume iterations without recovery.

---

## Plan-and-Execute

**Entry point:** `ovos-plan-execute-loop`
**Class:** `PlanAndExecuteEngine` — `ovos_agentic_loop/plan_execute.py:108`
**Reference:** Wang et al., 2023 — *Plan-and-Solve Prompting*; also the
LangChain Plan-and-Execute agent pattern.

### How it works

Planning and execution are **two separate LLM calls** per run.

**Phase 1 — Plan:** the planner LLM receives the user's request and the full
tool list, then outputs a numbered list of 3–7 sub-tasks.

```
1. Get current weather in Paris
2. Get current weather in London
3. Compare and answer which is warmer
```

**Phase 2 — Execute:** each step runs through a mini-ReAct sub-loop
(`max_step_iterations`, default 5).  The output of every completed step is
appended as context before the next step starts.

**Phase 3 — Synthesize:** a single "summarise all step results" LLM call
produces the natural-language final answer.

### Loop logic (`plan_execute.py:246–298`)

```
plan = planner_llm(messages + tool_schemas)
steps = parse_numbered_list(plan)

step_results = []
for step in steps[:max_steps]:
    result = mini_react_loop(step, completed_so_far, tool_schemas)
    step_results.append(result)

answer = synthesizer_llm(original_request, step_results)
```

### When to use

- Requests that naturally decompose into **independent sub-goals** (e.g.
  "get weather in Paris and London, then compare").
- Workflows where the full plan must be visible before execution starts
  (e.g. for logging or human review).
- Tasks with **more than ~3 tool calls** — ReAct tends to lose track of
  earlier observations; Plan-and-Execute keeps step outputs explicit.

### Strengths and limits

**Strengths:** The planner phase produces an auditable, inspectable plan.
Step outputs are explicit and reusable.
**Limits:** Costs more LLM calls (1 planner + N executors + 1 synthesizer).
The plan is fixed after phase 1; if step 2 reveals the plan was wrong, the
engine cannot replan mid-execution.

---

## Reflexion — Self-Reflective Episodic Loop

**Entry point:** `ovos-reflexion-loop`
**Class:** `ReflexionEngine` — `ovos_agentic_loop/reflexion.py:82`
**Paper:** Shinn et al., 2023 — *Reflexion: Language Agents with Verbal
Reinforcement Learning*

### How it works

Reflexion adds an **outer episode loop** around ReAct.  After each episode
the brain evaluates its own answer.  If unsatisfactory, it generates a
concise verbal critique (*reflection*) that is prepended to the next
episode's system prompt.

```
Episode 1:
  inner ReAct → answer A
  Evaluator: UNSATISFACTORY — did not use the weather tool.
  Reflector: "Reflection: I answered from memory instead of calling
              get_current_weather. Next time I must use the tool."

Episode 2 (with reflection in context):
  inner ReAct → answer B
  Evaluator: SATISFACTORY — answer is complete.
  → return B
```

Iteration stops at `SATISFACTORY` or `max_reflections` (default 3).

### Loop logic (`reflexion.py:187–237`)

```
reflections = []

for episode in range(max_reflections):
    messages = prepend_reflections(reflections) + original_messages
    answer = inner_react.continue_chat(messages)
    ok, feedback = evaluate(original_request, answer)
    if ok:  → return answer
    if not last_episode:
        reflections.append(reflect(original_request, answer, feedback))

return last_answer  # best effort
```

The inner `ReActLoopEngine` shares the same `brain` and `toolboxes` as the
outer `ReflexionEngine`.  `set_brain()` propagates to both.

### When to use

- Tasks where a **wrong first attempt is likely** and recovery is cheap
  (e.g. coding, arithmetic, constrained-slot filling).
- Situations where the agent has several plausible approaches — reflections
  steer it away from already-failed strategies.
- When you want **automatic retry with diagnosis** without building explicit
  retry logic in the caller.

### Strengths and limits

**Strengths:** Often reaches a correct answer in 2 episodes that ReAct
would never recover from in one.  No external memory store — reflections
live in the prompt context.
**Limits:** Each episode is a full ReAct run; total LLM calls = episodes ×
ReAct iterations + evaluations + reflections.  The evaluator can be wrong
(false UNSATISFACTORY → unnecessary retries; false SATISFACTORY → early
exit with wrong answer).

---

## Self-Ask — Compositional Question Decomposition

**Entry point:** `ovos-self-ask-loop`
**Class:** `SelfAskEngine` — `ovos_agentic_loop/self_ask.py:112`
**Paper:** Press et al., 2022 — *Measuring and Narrowing the Compositionality
Gap in Language Models*

### How it works

The LLM decomposes a complex question into a chain of simpler follow-up
questions, each answered (typically via search) before the next is asked.

```
Question: Who is the president of the country that won FIFA World Cup 2022?
Are follow up questions needed here? Yes.
Follow up: Which country won FIFA World Cup 2022?
Intermediate answer: Argentina.
Follow up: Who is the president of Argentina?
Intermediate answer: Javier Milei.
So the final answer is: Javier Milei.
```

The grammar is intentionally simpler than ReAct — no `Action Input` JSON,
just a plain text query forwarded to the first available tool.

### Loop logic (`self_ask.py:255–328`)

```
for _ in range(max_follow_ups):
    response = brain.continue_chat(loop_messages)

    if "So the final answer is:" in response:  → extract and return
    if "Tool: X\nTool Input: Q" in response:   → call_named_tool(X, Q)
    if "Follow up: Q" in response and tools:   → call_first_tool(Q)
    else:                                       → return response as-is

    append (assistant: response) + (user: "Intermediate answer: {result}")
```

Without tools the engine still works as a pure-LLM chain-of-thought
decomposer: the LLM answers each follow-up from its own knowledge.

### When to use

- **Multi-hop knowledge questions** where each intermediate fact is
  independently look-up-able (e.g. "What language is spoken in the capital
  of the country that borders X?").
- Pipelines with a **single search/lookup tool** — the Self-Ask grammar is
  optimised for a simple `query → result` tool interface rather than
  multi-argument JSON tools.
- Situations where you want **explicit intermediate reasoning** visible in
  the transcript (useful for debugging or citation).

### Strengths and limits

**Strengths:** Very readable traces.  Works with zero tools (pure LLM
reasoning) or one search tool.  Simple grammar means small/weaker LLMs
follow the format more reliably than ReAct's JSON Action Input.
**Limits:** Poor fit for tasks requiring multi-argument tools or side
effects (write file, run command).  All sub-questions are answered
sequentially — no parallelism.  Cannot reuse an intermediate answer for
multiple downstream questions without the LLM re-asking.

---

## Comparison table

| Property | ReAct | Plan-and-Execute | Reflexion | Self-Ask |
| :--- | :---: | :---: | :---: | :---: |
| LLM calls per turn (min) | 1–N | 3+N | 2–(3+N)×E | 1–N |
| Supports multi-arg tools | ✓ | ✓ | ✓ | partial |
| Can self-correct | ✗ | ✗ | ✓ | ✗ |
| Produces auditable plan | ✗ | ✓ | ✗ | ✓ (trace) |
| Works without tools | ✓ | ✓ | ✓ | ✓ |
| Best for | general | multi-step workflows | correctness | multi-hop QA |

*E = episodes; N = tool calls per episode/step.*

---

## Composing loops

All four engines are standard `ChatEngine` / `AgenticLoopEngine` subclasses.
You can nest them or wrap them in any persona config:

```json
{
  "solvers": ["ovos-reflexion-loop"],
  "plugin-config": {
    "ovos-reflexion-loop": {
      "brain": "ovos-chat-openai-plugin",
      "max_reflections": 2,
      "max_iterations": 8,
      "toolboxes": ["ovos-web-search-tools", "ovos-filesystem-tools"]
    }
  }
}
```

The `ReflexionEngine` will internally build a `ReActLoopEngine` configured
with the same `brain` and `toolboxes`; no extra wiring is required.
