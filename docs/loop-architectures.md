# Agent Loop Architectures

`ovos-agentic-loop` ships seven distinct loop strategies. Each is a concrete
`AgenticLoopEngine` (`ovos_agentic_loop/base.py:8`) and is registered as an
`opm.agents.chat` entry point so `ovos-persona` can load it by ID.

---

## Choosing a loop

| Use-case signal | Recommended loop |
| :--- | :--- |
| No tools, pure reasoning / arithmetic / logic | **Chain-of-Thought** |
| Single-turn tool use, general assistant | **ReAct** |
| Multi-step task with distinct, sequenced sub-goals | **Plan-and-Execute** |
| Correctness matters; agent may fail on first attempt | **Reflexion** |
| Multi-hop knowledge question (chain of facts) | **Self-Ask** |
| Answer contains verifiable factual claims | **CRITIC** |
| Multiple solution strategies exist; want best one | **Tree-of-Thoughts** |

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

## Chain-of-Thought — Structured Step-by-Step Reasoning

**Entry point:** `ovos-chain-of-thought-loop`
**Class:** `ChainOfThoughtEngine` — `ovos_agentic_loop/chain_of_thought.py:68`
**Papers:** Wei et al., 2022 — *Chain-of-Thought Prompting Elicits Reasoning in
Large Language Models*; Kojima et al., 2022 — *Large Language Models are
Zero-Shot Reasoners* ("Let's think step by step")

### How it works

A single LLM call with a system prompt instructing the model to reason step by
step before committing to a final answer.  The ``FINAL ANSWER:`` marker is
extracted from the structured response.

```
Step 1: 17 × 6 = 102
Step 2: 102 + 14 = 116
FINAL ANSWER: 116
```

No tools, no loop, no iteration.

### Loop logic (`chain_of_thought.py:118–148`)

```
messages = [cot_system_prompt + optional_extra_prompt] + conversation_history
response = brain.continue_chat(messages)
return extract_after("FINAL ANSWER:", response) or response
```

### When to use

- **Arithmetic and algebra** — multi-step calculation where intermediate
  values matter.
- **Logic puzzles and constraint solving** — tasks that require eliminating
  cases.
- **Multi-step instruction following** — decomposing "how to do X" questions.
- Any task where **no external information** is needed; adding tools would
  add latency with no benefit.
- As a cheap **first pass** before escalating to a more expensive loop.

### Strengths and limits

**Strengths:** Exactly one LLM call, lowest latency and cost of all seven
loops.  Readable reasoning trace in the response.  Zero dependencies on tools
or external services.
**Limits:** Hallucination-prone on factual questions — the model reasons from
its training data only.  Does not retry or self-correct.

---

## CRITIC — Tool-Assisted Self-Verification and Revision

**Entry point:** `ovos-critic-loop`
**Class:** `CRITICEngine` — `ovos_agentic_loop/critic.py:92`
**Paper:** Gou et al., 2023 — *CRITIC: Large Language Models Can Self-Correct
with Tool-Interactive Critiquing*

### How it works

Three phases: **draft → critique → revise**.

1. **Draft**: the brain generates an initial answer.
2. **Critique**: a separate LLM call identifies verifiable claims in the draft
   and emits ``CLAIM / TOOL / TOOL INPUT`` blocks for each.
3. **Verify + Revise**: each claim is checked via a tool call; observations
   are used to rewrite the answer.  Repeats up to ``max_critique_rounds``.

```
Draft:    "The Eiffel Tower was built in 1887."
Critique: CLAIM: Built in 1887
          TOOL: web_search
          TOOL INPUT: Eiffel Tower construction year
Verify:   → "Construction 1887–1889; opened 1889."
Revised:  "The Eiffel Tower was built between 1887 and 1889."
```

If the brain emits ``VERIFIED: all claims are correct`` the draft is accepted
without revision.

### Loop logic (`critic.py:198–260`)

```
draft = brain(messages)
if no tools: return draft

for _ in range(max_critique_rounds):
    critique = brain("critique: " + draft)
    if VERIFIED in critique: break
    blocks = parse_claim_tool_blocks(critique)
    if not blocks: break
    verifications = [call_tool(b.tool, b.tool_input) for b in blocks]
    draft = brain("revise with: " + verifications)
return draft
```

### When to use

- **Factual Q&A** where the answer may contain specific numbers, dates,
  names, or statistics that can be checked with a search tool.
- Pipelines where **accuracy is more important than latency** and a single
  draft pass is not trustworthy enough.
- Tasks where Reflexion would be overkill (full re-runs) but a lightweight
  fact-check pass is sufficient.

### Strengths and limits

**Strengths:** Targets errors precisely at the claim level — only
incorrect facts are revised, the rest of the answer is preserved.  More
efficient than Reflexion for factual corrections (no full re-run).
**Limits:** Requires a capable LLM to produce well-formed ``CLAIM/TOOL/TOOL
INPUT`` blocks reliably.  Works poorly on subjective or opinion questions
where there is nothing to verify.  With no tools registered, falls back to
draft-only (equivalent to a simple brain call).

---

## Tree-of-Thoughts — Beam Search over Reasoning Paths

**Entry point:** `ovos-tree-of-thoughts-loop`
**Class:** `TreeOfThoughtsEngine` — `ovos_agentic_loop/tree_of_thoughts.py:108`
**Paper:** Yao et al., 2023 — *Tree of Thoughts: Deliberate Problem Solving
with Large Language Models*

### How it works

At each depth level the engine generates ``n_branches`` independent candidate
reasoning steps, scores each one with a separate evaluator LLM call, and
keeps only the top ``beam_width`` branches for the next level (beam search).

```
Depth 0 (root):
  Branch A: "Try approach X …"   score 4
  Branch B: "Try approach Y …"   score 9  ← kept (beam_width=1)
  Branch C: "Try approach Z …"   score 2

Depth 1 (from B):
  Branch B1: "Refine Y …"   score 7  ← kept
  Branch B2: "Try Z next …" score 3
  Branch B3: ANSWER: 42     → return immediately
```

Any branch that produces ``ANSWER:`` in a generated thought terminates the
search immediately.  If ``max_depth`` is reached without a natural answer,
the highest-scored surviving branch is asked to produce a final answer.

### Loop logic (`tree_of_thoughts.py:211–264`)

```
branches = [_Branch()]  # single empty root

for depth in range(max_depth):
    candidates = []
    for branch in branches:
        for _ in range(n_branches):
            thought = generate(problem, branch)
            if ANSWER in thought: return answer
            score = evaluate(problem, branch, thought)
            candidates.append((branch + thought, score))

    # keep top beam_width by score
    branches = sorted(candidates, key=score)[:beam_width]

return force_answer(best_branch)
```

LLM calls per run: ``depth × n_branches × 2`` (generator + evaluator per
branch per level) + 1 optional force_answer.

### When to use

- Problems with **multiple competing solution strategies** where it is not
  clear upfront which approach will work (combinatorics, coding, planning).
- Tasks where **early commitment** (as in ReAct) leads to dead ends — ToT
  can explore and abandon a branch before committing to it.
- **Creative tasks** (writing, brainstorming) where you want the LLM to
  generate diverse options and keep the best.

### Strengths and limits

**Strengths:** Can recover from locally-plausible but globally-poor choices
by keeping competing branches alive.  The evaluator provides an explicit
quality signal at each step.
**Limits:** Most expensive of the seven loops — LLM call count grows as
``depth × n_branches × 2``.  The evaluator itself can be biased or wrong.
Only BFS/beam-search is implemented; DFS with backtracking is not (context
window cost).

---

## Comparison table

| Property | CoT | ReAct | Plan+Exec | Reflexion | Self-Ask | CRITIC | ToT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LLM calls (min) | 1 | 1–N | 3+N | 2–(3+N)×E | 1–N | 2 | d×b×2 |
| Supports multi-arg tools | ✗ | ✓ | ✓ | ✓ | partial | partial | ✗ |
| Can self-correct | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | partial |
| Produces auditable plan/trace | ✓ | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ |
| Works without tools | ✓ | ✓ | ✓ | ✓ | ✓ | ✓* | ✓ |
| Best for | reasoning | general | multi-step | correctness | multi-hop QA | factual Q&A | hard problems |

*CRITIC without tools skips critique phases and returns the draft directly.*
*CoT = Chain-of-Thought; E = episodes; N = tool calls; d = depth; b = n_branches.*

---

## Composing loops

All seven engines are standard `ChatEngine` / `AgenticLoopEngine` subclasses.
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
