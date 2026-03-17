# SUGGESTIONS — ovos-agentic-loop

- **PlanExecuteLoopEngine**: Add `plan_execute.py` implementing a two-phase loop (plan all steps, execute sequentially). Declared in the plan as future work.
- **Auto brain sharing**: `ReActLoopEngine` could automatically call `set_brain()` on any `SkillMDToolBox` in its toolbox list, removing manual wiring.
- **Async variant**: `AsyncAgenticLoopEngine` with `async continue_chat` for non-blocking tool dispatch.
- **Streaming support**: Hook into `ChatEngine.stream_*` methods for partial result streaming during the loop.
- **SKILL.md entry-point helper**: Add a `register_skill_md(path)` utility that generates the correct pyproject.toml snippet for package authors.
