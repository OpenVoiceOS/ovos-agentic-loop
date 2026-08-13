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
"""ChainOfThoughtEngine — structured step-by-step reasoning without tool calls.

Based on "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
(Wei et al., 2022 — https://arxiv.org/abs/2201.11903) and the follow-up
zero-shot variant "Large Language Models are Zero-Shot Reasoners" (Kojima et al.,
2022 — https://arxiv.org/abs/2205.11916).

Algorithm
---------
A single LLM call is made with a system prompt that instructs the model to
reason step by step before giving its final answer.  The final answer is
extracted from a ``FINAL ANSWER:`` marker; if absent the full response is
returned.

This is the **simplest possible agent loop** — one LLM call, no tools, no
iteration.  It is the recommended baseline for reasoning-heavy tasks that do
not require external information, and it is the inner building block used by
all more complex engines.

Key differences from ReAct
--------------------------
- No tools, no observation loop.
- Single LLM call — fastest and cheapest.
- Produces human-readable reasoning traces naturally.
- Best for arithmetic, logic, common-sense reasoning, multi-step instructions.
"""
from typing import Any, Dict, List, Optional

from ovos_plugin_manager.templates.agents import AgentMessage, ChatEngine, MessageRole, ToolsArg

from ovos_agentic_loop.base import AgenticLoopEngine

_COT_SYSTEM_PROMPT = """\
Think through the problem step by step before giving your final answer.

Format your response as:
Step 1: <reasoning>
Step 2: <reasoning>
...
FINAL ANSWER: <concise answer>

Rules:
- Work through every relevant step explicitly.
- Do not skip steps.
- Put ONLY the final answer after "FINAL ANSWER:" — no extra reasoning.
"""

_FINAL_ANSWER_MARKER = "FINAL ANSWER:"


def _extract_final_answer(text: str) -> Optional[str]:
    """
    Extract the text following ``FINAL ANSWER:`` in LLM output.

    Args:
        text: Raw LLM output.

    Returns:
        The answer string, or ``None`` if the marker is absent.
    """
    idx = text.upper().find(_FINAL_ANSWER_MARKER.upper())
    if idx == -1:
        return None
    return text[idx + len(_FINAL_ANSWER_MARKER):].strip()


class ChainOfThoughtEngine(AgenticLoopEngine):
    """
    ``AgenticLoopEngine`` implementing zero-shot Chain-of-Thought prompting.

    Adds a "think step by step" system prompt to every request and extracts
    the ``FINAL ANSWER:`` from the structured response.  No tools, no loop —
    a single LLM call per ``continue_chat`` invocation.

    This is the recommended baseline for reasoning tasks (arithmetic, logic,
    multi-step instructions) that do not require external information.
    Registered toolboxes are ignored.

    Config keys:

    - ``brain`` (str): ChatEngine plugin ID used as the inner LLM.
    - ``system_prompt`` (str): Optional extra system context prepended before
      the CoT instruction.

    Entry point group: ``opm.agents.chat``
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialise the Chain-of-Thought engine.

        Args:
            config: Plugin configuration dict.
        """
        super().__init__(config=config)
        self._brain: Optional[ChatEngine] = None

    @property
    def brain(self) -> Optional[ChatEngine]:
        """The inner ChatEngine used for the single LLM call."""
        if self._brain is None:
            self._brain = self._load_brain()
        return self._brain

    def set_brain(self, brain: ChatEngine) -> None:
        """
        Inject a ChatEngine instance as the inner LLM.

        Args:
            brain: Instantiated ``ChatEngine``.
        """
        self._brain = brain
        self._inject_brain_into_toolboxes(brain)

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

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None,
                      tools: "ToolsArg" = None) -> AgentMessage:
        """
        Run a single CoT-prompted LLM call and return the final answer.

        Prepends the CoT system instruction (and any ``system_prompt`` config
        value) to the message list, calls the brain once, and extracts the
        ``FINAL ANSWER:`` text.  If the marker is absent the full response is
        returned as-is.

        Args:
            messages: Conversation history including the latest user turn.
            session_id: Session identifier forwarded to the brain.
            lang: BCP-47 language code forwarded to the brain.
            units: Measurement system forwarded to the brain.

        Returns:
            ``AgentMessage`` with ``MessageRole.ASSISTANT`` containing the
            extracted final answer or the full CoT response.
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

        extra_prompt = self.config.get("system_prompt", "")
        system_content = (extra_prompt + "\n\n" if extra_prompt else "") + _COT_SYSTEM_PROMPT

        loop_messages = [
            AgentMessage(role=MessageRole.SYSTEM, content=system_content),
            *messages,
        ]
        response = self.brain.continue_chat(
            loop_messages, session_id=session_id, lang=lang, units=units
        )
        final = _extract_final_answer(response.content)
        content = final if final is not None else response.content
        return AgentMessage(role=MessageRole.ASSISTANT, content=content)
