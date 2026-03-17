# Copyright 2025, OpenVoiceOS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PlanAndExecuteEngine — two-phase agent loop: plan first, then execute each step.

Based on "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning
by Large Language Models" (Wang et al., 2023) and the LangChain Plan-and-Execute
agent pattern.

Algorithm
---------
1. **Plan phase**: a *planner* LLM call receives the user's request and the list of
   available tools, and outputs a numbered list of sub-tasks.
2. **Execute phase**: an *executor* LLM (can be the same brain) processes each
   sub-task in order using ReAct-style tool calls.  The output of each step is
   appended as context before the next step runs.
3. **Synthesize phase**: after all steps are executed the brain composes a final
   natural-language answer from the accumulated step results.

Key differences from ReAct
--------------------------
- Planning and execution are **separate LLM calls** — the planner decides *what*
  to do before any tools are called.
- Individual steps may each invoke multiple tools via a mini-ReAct sub-loop.
- The final answer is synthesized from step outputs rather than emerging
  incrementally.
"""
import json
import re
from typing import Any, Dict, List, Optional

from ovos_plugin_manager.templates.agents import AgentMessage, ChatEngine, MessageRole

from ovos_agentic_loop.base import AgenticLoopEngine
from ovos_agentic_loop.react import _build_react_system, _extract_final_answer, _parse_action

_PLANNER_SYSTEM_PROMPT = """\
You are a planning assistant.  Given a user request and a list of available tools,
create a concise numbered action plan.

Rules:
- Output ONLY the numbered plan, one step per line.
- Each step should be a clear, actionable sub-task.
- Reference tool names where relevant (e.g. "Call get_current_weather for Paris").
- Do NOT execute anything — only plan.
- Use 3–7 steps maximum.

Available tools:
{tool_schemas}
"""

_EXECUTOR_STEP_PROMPT = """\
You are executing step {step_num} of a multi-step plan.

Overall plan:
{plan}

Steps completed so far:
{completed}

Current step to execute:
{step}

Use the available tools as needed, then report the result of this step.
When done, output: STEP_RESULT: <brief summary of what you found/did>
"""

_SYNTHESIZER_PROMPT = """\
You have executed all steps of a plan.  Synthesize a clear, complete answer
for the user based on the step results below.

Original request: {original_request}

Step results:
{step_results}

Write a natural, conversational reply to the user.
"""


def _parse_plan(text: str) -> List[str]:
    """
    Extract numbered steps from a planner LLM response.

    Args:
        text: Raw LLM output from the planner.

    Returns:
        List of step strings in order.
    """
    steps: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        match = re.match(r"^\d+[\.\)]\s+(.+)", line)
        if match:
            steps.append(match.group(1).strip())
    return steps


def _extract_step_result(text: str) -> Optional[str]:
    """
    Extract the ``STEP_RESULT:`` summary from an executor LLM response.

    Args:
        text: Raw LLM output from the executor.

    Returns:
        The result string, or ``None`` if the token was not found.
    """
    idx = text.find("STEP_RESULT:")
    if idx == -1:
        return None
    return text[idx + len("STEP_RESULT:"):].strip()


class PlanAndExecuteEngine(AgenticLoopEngine):
    """
    ``AgenticLoopEngine`` implementing the Plan-and-Execute pattern.

    The brain LLM is called in three phases: **plan**, **execute each step**
    (with optional tool calls), and **synthesize** the final answer.

    Config keys:

    - ``brain`` (str): ChatEngine plugin ID used as planner/executor/synthesizer.
    - ``toolboxes`` (List[str]): ToolBox plugin IDs to load (inherited).
    - ``max_step_iterations`` (int): Max tool-call cycles per step (default: 5).
    - ``max_steps`` (int): Maximum plan steps to execute (default: 10).

    Entry point group: ``opm.agents.chat``
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialise the Plan-and-Execute engine.

        Args:
            config: Plugin configuration dict.
        """
        super().__init__(config=config)
        self._brain: Optional[ChatEngine] = None

    @property
    def max_step_iterations(self) -> int:
        """Maximum tool-call cycles within a single step's executor sub-loop."""
        return int(self.config.get("max_step_iterations", 5))

    @property
    def max_steps(self) -> int:
        """Maximum number of plan steps to execute."""
        return int(self.config.get("max_steps", 10))

    @property
    def brain(self) -> Optional[ChatEngine]:
        """The inner ChatEngine used for all LLM phases."""
        if self._brain is None:
            self._brain = self._load_brain()
        return self._brain

    def set_brain(self, brain: ChatEngine) -> None:
        """
        Inject a ChatEngine instance as the inner LLM.

        Args:
            brain: Instantiated ``ChatEngine`` to use for all LLM calls.
        """
        self._brain = brain

    def _load_brain(self) -> Optional[ChatEngine]:
        """
        Load the brain ChatEngine from config using OPM.

        Returns:
            Instantiated ``ChatEngine``, or ``None`` if loading fails.
        """
        brain_id: str = self.config.get("brain", "")
        if not brain_id:
            return None
        try:
            from ovos_plugin_manager.agents import load_chat_plugin
            return load_chat_plugin(brain_id, config=self.config.get(brain_id, {}))
        except Exception:  # noqa: BLE001
            return None

    def _collect_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Gather JSON schemas from all registered toolboxes.

        Returns:
            Flat list of tool schema dicts.
        """
        schemas: List[Dict[str, Any]] = []
        for tb in self.toolboxes:
            try:
                schemas.extend(tb.tool_json_list)
            except Exception:  # noqa: BLE001
                pass
        return schemas

    def _call_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Execute a named tool across all registered toolboxes.

        Args:
            tool_name: Name of the tool to invoke.
            args: Keyword arguments to pass to the tool.

        Returns:
            String representation of the tool output.
        """
        for tb in self.toolboxes:
            try:
                tool = tb.get_tool(tool_name)
                if tool is not None:
                    result = tb.call_tool(tool_name, args)
                    return str(result)
            except Exception:  # noqa: BLE001
                continue
        return f"Error: tool '{tool_name}' not found."

    def _execute_step(self, step: str, plan: str, completed: str,
                      step_num: int, tool_schemas: List[Dict[str, Any]],
                      session_id: str, lang: Optional[str],
                      units: Optional[str]) -> str:
        """
        Execute a single plan step using a mini ReAct sub-loop.

        Args:
            step: The step description to execute.
            plan: Full plan text for context.
            completed: Summary of previously completed steps.
            step_num: 1-based step number.
            tool_schemas: Available tool schemas.
            session_id: Session identifier for the brain.
            lang: BCP-47 language code.
            units: Measurement system.

        Returns:
            The ``STEP_RESULT`` summary extracted from the executor's output,
            or the raw last LLM response if no token was found.
        """
        system_content = _build_react_system(tool_schemas) if tool_schemas else ""
        user_content = _EXECUTOR_STEP_PROMPT.format(
            step_num=step_num,
            plan=plan,
            completed=completed or "(none yet)",
            step=step,
        )
        messages: List[AgentMessage] = []
        if system_content:
            messages.append(AgentMessage(role=MessageRole.SYSTEM, content=system_content))
        messages.append(AgentMessage(role=MessageRole.USER, content=user_content))

        last_text = ""
        for _ in range(self.max_step_iterations):
            response = self.brain.continue_chat(
                messages, session_id=session_id, lang=lang, units=units
            )
            last_text = response.content

            result = _extract_step_result(last_text)
            if result is not None:
                return result

            # Check if the executor emitted a tool action.
            parsed = _parse_action(last_text)
            if parsed is None:
                break

            tool_name, args = parsed
            observation = self._call_tool(tool_name, args)
            messages.append(AgentMessage(role=MessageRole.ASSISTANT, content=last_text))
            messages.append(AgentMessage(
                role=MessageRole.USER,
                content=f"Observation: {observation}",
            ))

        return _extract_step_result(last_text) or last_text

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        """
        Run the Plan-and-Execute loop and return the final response.

        Phases: (1) call planner, (2) execute each step, (3) synthesize answer.

        Args:
            messages: Conversation history including the latest user turn.
            session_id: Session identifier forwarded to the brain.
            lang: BCP-47 language code forwarded to the brain.
            units: Measurement system forwarded to the brain.

        Returns:
            ``AgentMessage`` with ``MessageRole.ASSISTANT`` containing the
            synthesized final answer.
        """
        if self.brain is None:
            return AgentMessage(role=MessageRole.ASSISTANT,
                                content="Error: no brain configured.")

        tool_schemas = self._collect_tool_schemas()

        # Extract the user's latest message for the synthesizer.
        original_request = ""
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                original_request = msg.content
                break

        # --- Phase 1: Plan ---
        planner_messages = [
            AgentMessage(
                role=MessageRole.SYSTEM,
                content=_PLANNER_SYSTEM_PROMPT.format(
                    tool_schemas=json.dumps(tool_schemas, indent=2)
                ),
            ),
            *messages,
        ]
        plan_response = self.brain.continue_chat(
            planner_messages, session_id=session_id, lang=lang, units=units
        )
        plan_text = plan_response.content
        steps = _parse_plan(plan_text)

        if not steps:
            # Planner didn't produce a parseable plan — fall back to direct answer.
            return AgentMessage(role=MessageRole.ASSISTANT, content=plan_text)

        steps = steps[: self.max_steps]

        # --- Phase 2: Execute each step ---
        step_results: List[str] = []
        for i, step in enumerate(steps, start=1):
            completed_summary = "\n".join(
                f"Step {j + 1}: {r}" for j, r in enumerate(step_results)
            )
            result = self._execute_step(
                step=step,
                plan=plan_text,
                completed=completed_summary,
                step_num=i,
                tool_schemas=tool_schemas,
                session_id=session_id,
                lang=lang,
                units=units,
            )
            step_results.append(result)

        # --- Phase 3: Synthesize ---
        step_results_text = "\n".join(
            f"Step {i + 1} ({steps[i]}): {r}" for i, r in enumerate(step_results)
        )
        synth_messages = [
            AgentMessage(
                role=MessageRole.USER,
                content=_SYNTHESIZER_PROMPT.format(
                    original_request=original_request,
                    step_results=step_results_text,
                ),
            )
        ]
        final_response = self.brain.continue_chat(
            synth_messages, session_id=session_id, lang=lang, units=units
        )
        return AgentMessage(role=MessageRole.ASSISTANT, content=final_response.content)
