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
"""ReflexionEngine — ReAct + verbal self-reflection between episodes.

Based on "Reflexion: Language Agents with Verbal Reinforcement Learning"
(Shinn et al., 2023 — https://arxiv.org/abs/2303.11366).

Algorithm
---------
Each *episode* runs a complete inner ReAct loop.  After the episode the
evaluator LLM (same brain) determines whether the answer is satisfactory.
If not, a *reflector* LLM call generates a concise verbal critique that is
stored as ``past_reflections``.  On the next episode the reflections are
prepended to the system prompt as "lessons learned", steering the agent away
from previous mistakes.

Iteration stops when:
- The evaluator judges the answer satisfactory, or
- ``max_reflections`` episodes have been exhausted (the last answer is returned).

Key differences from ReAct
--------------------------
- Adds a **self-evaluation** step after every episode.
- Failed attempts produce a **verbal reflection** kept in episodic memory.
- Subsequent episodes are conditioned on accumulated reflections.
- No external memory store — reflections live in the prompt context.
"""
from typing import Any, Dict, List, Optional

from ovos_plugin_manager.templates.agents import AgentMessage, ChatEngine, MessageRole, ToolsArg

from ovos_agentic_loop.base import AgenticLoopEngine
from ovos_agentic_loop.react import ReActLoopEngine

_EVALUATOR_PROMPT = """\
You are evaluating whether an AI assistant's answer fully and correctly satisfies
the user's request.

User request: {request}

Assistant answer: {answer}

Reply with exactly one of:
  SATISFACTORY   — if the answer is complete, accurate, and directly addresses the request.
  UNSATISFACTORY — if the answer is incomplete, incorrect, or evasive.

Then on the next line briefly explain why (one sentence).
"""

_REFLECTOR_PROMPT = """\
You are a self-reflection assistant.  The AI agent below attempted to answer a
user's request but the answer was judged unsatisfactory.

User request: {request}

Agent's answer: {answer}

Evaluator feedback: {feedback}

Write a concise reflection (2–4 sentences) that will help the agent do better on
the next attempt.  Focus on:
- What went wrong (wrong tool used, missing information, incorrect reasoning).
- What to do differently next time.

Begin your reflection with "Reflection: ".
"""

_REFLEXION_SYSTEM_PREFIX = """\
You have attempted this task before and reflected on your previous mistakes.
Apply these lessons to do better this time:

{reflections}

"""


class ReflexionEngine(AgenticLoopEngine):
    """
    ``AgenticLoopEngine`` implementing the Reflexion pattern.

    Wraps an inner :class:`~ovos_agentic_loop.react.ReActLoopEngine` with an
    outer episode loop.  After each episode the brain evaluates its own answer
    and, if unsatisfied, generates a verbal self-reflection.  Subsequent
    episodes are primed with accumulated reflections.

    Config keys:

    - ``brain`` (str): ChatEngine plugin ID used as inner LLM.
    - ``toolboxes`` (List[str]): ToolBox plugin IDs to load (inherited).
    - ``max_reflections`` (int): Maximum episodes / reflection cycles (default: 3).
    - ``max_iterations`` (int): Max ReAct iterations per episode (default: 10).

    Entry point group: ``opm.agents.chat``
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialise the Reflexion engine.

        Args:
            config: Plugin configuration dict.
        """
        super().__init__(config=config)
        self._brain: Optional[ChatEngine] = None
        self._inner_react: Optional[ReActLoopEngine] = None

    @property
    def max_reflections(self) -> int:
        """Maximum number of reflection episodes before accepting the last answer."""
        return int(self.config.get("max_reflections", 3))

    @property
    def brain(self) -> Optional[ChatEngine]:
        """The inner ChatEngine used for all LLM phases."""
        if self._brain is None:
            self._brain = self._load_brain()
        return self._brain

    def set_brain(self, brain: ChatEngine) -> None:
        """
        Inject a ChatEngine instance as the inner LLM.

        Also injects it into the inner ReAct engine.

        Args:
            brain: Instantiated ``ChatEngine`` to use for all LLM calls.
        """
        self._brain = brain
        self._inject_brain_into_toolboxes(brain)
        if self._inner_react is not None:
            self._inner_react.set_brain(brain)

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

    def _get_react_engine(self) -> ReActLoopEngine:
        """
        Return (or lazily create) the inner ReAct engine.

        Returns:
            Configured :class:`ReActLoopEngine` sharing the same brain and toolboxes.
        """
        if self._inner_react is None:
            inner_config = dict(self.config)
            inner_config["max_iterations"] = self.config.get("max_iterations", 10)
            self._inner_react = ReActLoopEngine(config=inner_config)
            if self.brain is not None:
                self._inner_react.set_brain(self.brain)
            self._inner_react.toolboxes = self.toolboxes
        return self._inner_react

    def _evaluate(self, request: str, answer: str,
                  session_id: str, lang: Optional[str],
                  units: Optional[str]) -> tuple[bool, str]:
        """
        Ask the brain to evaluate whether the answer is satisfactory.

        Args:
            request: The original user request.
            answer: The agent's latest answer.
            session_id: Session identifier.
            lang: BCP-47 language code.
            units: Measurement system.

        Returns:
            ``(is_satisfactory, feedback_text)`` tuple.
        """
        prompt = _EVALUATOR_PROMPT.format(request=request, answer=answer)
        response = self.brain.continue_chat(
            [AgentMessage(role=MessageRole.USER, content=prompt)],
            session_id=session_id, lang=lang, units=units,
        )
        text = response.content.strip()
        is_ok = text.upper().startswith("SATISFACTORY") and not text.upper().startswith("UNSATISFACTORY")
        return is_ok, text

    def _reflect(self, request: str, answer: str, feedback: str,
                 session_id: str, lang: Optional[str],
                 units: Optional[str]) -> str:
        """
        Generate a verbal self-reflection on a failed episode.

        Args:
            request: The original user request.
            answer: The agent's latest answer.
            feedback: Evaluator feedback text.
            session_id: Session identifier.
            lang: BCP-47 language code.
            units: Measurement system.

        Returns:
            Reflection string beginning with ``"Reflection: "``.
        """
        prompt = _REFLECTOR_PROMPT.format(
            request=request, answer=answer, feedback=feedback
        )
        response = self.brain.continue_chat(
            [AgentMessage(role=MessageRole.USER, content=prompt)],
            session_id=session_id, lang=lang, units=units,
        )
        return response.content.strip()

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None,
                      tools: "ToolsArg" = None) -> AgentMessage:
        """
        Run the Reflexion loop and return the final response.

        Executes up to ``max_reflections`` episodes.  Each episode runs the inner
        ReAct engine; the result is evaluated and, if unsatisfactory, reflected
        upon before the next episode.

        Args:
            messages: Conversation history including the latest user turn.
            session_id: Session identifier forwarded to the brain.
            lang: BCP-47 language code forwarded to the brain.
            units: Measurement system forwarded to the brain.

        Returns:
            ``AgentMessage`` with ``MessageRole.ASSISTANT`` containing the best
            answer found across all episodes.
        """
        # `tools` is accepted (and ignored) purely for contract conformance with
        # ovos_plugin_manager.templates.agents.ChatEngine.continue_chat, whose
        # signature declares it unconditionally. This engine is not tool-capable
        # (supports_tools stays False). Accepting the kwarg matters because the
        # agentic-loop ReAct fallback (see native_toolcall.py) calls
        # `self.brain.continue_chat(..., tools=...)` on whatever brain engine is
        # configured, even non-tool-capable ones — omitting `tools` here would
        # raise TypeError on that call path.
        if self.brain is None:
            return AgentMessage(role=MessageRole.ASSISTANT,
                                content="Error: no brain configured.")

        # Extract original request text.
        original_request = ""
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                original_request = msg.content
                break

        react = self._get_react_engine()
        reflections: List[str] = []
        last_answer = ""

        for episode in range(self.max_reflections):
            # Build episode messages: prepend reflections if any.
            episode_messages: List[AgentMessage] = list(messages)
            if reflections:
                reflection_context = _REFLEXION_SYSTEM_PREFIX.format(
                    reflections="\n".join(f"- {r}" for r in reflections)
                )
                episode_messages = [
                    AgentMessage(role=MessageRole.SYSTEM, content=reflection_context),
                    *episode_messages,
                ]

            # Run inner ReAct episode.
            result = react.continue_chat(
                episode_messages, session_id=session_id, lang=lang, units=units
            )
            last_answer = result.content

            # Evaluate the answer.
            is_ok, feedback = self._evaluate(
                original_request, last_answer, session_id, lang, units
            )
            if is_ok:
                return AgentMessage(role=MessageRole.ASSISTANT, content=last_answer)

            # Generate a reflection for the next episode (skip on last iteration).
            if episode < self.max_reflections - 1:
                reflection = self._reflect(
                    original_request, last_answer, feedback, session_id, lang, units
                )
                reflections.append(reflection)

        return AgentMessage(role=MessageRole.ASSISTANT, content=last_answer)
