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
"""SelfAskEngine — question decomposition via follow-up sub-questions.

Based on "Measuring and Narrowing the Compositionality Gap in Language Models"
(Press et al., 2022 — https://arxiv.org/abs/2210.03350).

Algorithm
---------
The LLM is prompted to determine whether follow-up questions are needed to
answer the user's question.  If yes, it generates the first follow-up question.
A *search tool* (or any registered tool) resolves the sub-question into an
*intermediate answer*.  The agent then decides whether more follow-up
questions are needed, continuing until it has all necessary facts, at which
point it produces the ``So the final answer is:`` synthesis.

Canonical trace (from the paper)
---------------------------------
::

    Question: Who is the president of the country that won FIFA 2022?
    Are follow up questions needed here? Yes.
    Follow up: Which country won FIFA World Cup 2022?
    Intermediate answer: Argentina.
    Follow up: Who is the president of Argentina?
    Intermediate answer: Javier Milei.
    So the final answer is: Javier Milei.

Key differences from ReAct
---------------------------
- Designed specifically for **multi-hop knowledge questions** that require
  combining several independently-answerable facts.
- Each sub-question is answered by calling exactly one tool (typically search).
- The loop grammar (``Follow up:`` / ``Intermediate answer:`` / ``So the final
  answer is:``) is simpler than ReAct's Thought/Action/Observation triplets.
- Works well with zero tools (pure LLM decomposition) or a single search tool.
"""
import re
from typing import Any, Dict, List, Optional

from ovos_plugin_manager.templates.agents import AgentMessage, ChatEngine, MessageRole

from ovos_agentic_loop.base import AgenticLoopEngine

_FINAL_ANSWER_PREFIX = "So the final answer is:"
_FOLLOW_UP_PREFIX = "Follow up:"
_INTERMEDIATE_PREFIX = "Intermediate answer:"

_SELF_ASK_SYSTEM_PROMPT = """\
Answer questions by decomposing them into simpler follow-up questions.

Rules:
1. First ask: "Are follow up questions needed here?"
2. If Yes: emit "Follow up: <sub-question>" — ONE sub-question at a time.
3. After each "Intermediate answer:" decide if more follow-ups are needed.
4. When you have enough information, emit:
   "So the final answer is: <answer>"

{tool_note}

Example:
Question: Who is the CEO of the company that makes the iPhone?
Are follow up questions needed here? Yes.
Follow up: Which company makes the iPhone?
Intermediate answer: Apple Inc.
Follow up: Who is the CEO of Apple Inc.?
Intermediate answer: Tim Cook.
So the final answer is: Tim Cook.
"""

_TOOL_NOTE_WITH_SEARCH = """\
To answer a follow-up question, call a search/lookup tool using EXACTLY this format:
  Tool: <tool_name>
  Tool Input: <query string>
Available tools: {tool_names}
"""

_TOOL_NOTE_NO_TOOLS = "(No external tools available — answer from your own knowledge.)"


def _extract_final_answer(text: str) -> Optional[str]:
    """
    Extract the answer following ``So the final answer is:`` in LLM output.

    Args:
        text: Raw LLM output text.

    Returns:
        The answer string, or ``None`` if the prefix was not present.
    """
    idx = text.lower().find(_FINAL_ANSWER_PREFIX.lower())
    if idx == -1:
        return None
    return text[idx + len(_FINAL_ANSWER_PREFIX):].strip()


def _extract_follow_up(text: str) -> Optional[str]:
    """
    Extract the sub-question following ``Follow up:`` in LLM output.

    Args:
        text: Raw LLM output text.

    Returns:
        The sub-question string, or ``None`` if not found.
    """
    match = re.search(r"Follow up:\s*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _extract_tool_call(text: str) -> Optional[tuple]:
    """
    Extract ``(tool_name, query)`` from a ``Tool:`` / ``Tool Input:`` block.

    Args:
        text: Raw LLM output text.

    Returns:
        ``(tool_name, query)`` or ``None`` if not found.
    """
    tool_match = re.search(r"Tool:\s*(\S+)", text, re.IGNORECASE)
    input_match = re.search(r"Tool Input:\s*(.+)", text, re.IGNORECASE)
    if tool_match and input_match:
        return tool_match.group(1).strip(), input_match.group(1).strip()
    return None


class SelfAskEngine(AgenticLoopEngine):
    """
    ``AgenticLoopEngine`` implementing the Self-Ask pattern.

    The brain LLM decomposes the user's question into follow-up sub-questions.
    Each sub-question is answered by calling a registered tool (typically a
    search/lookup tool) or, if no tools are available, answered from the LLM's
    own knowledge.  Decomposition continues until the LLM emits the final
    answer.

    This engine is most effective for **multi-hop knowledge questions** where
    each fact can be looked up independently.  Unlike ReAct it does not support
    complex multi-argument tool calls — each tool receives a plain text query.

    Config keys:

    - ``brain`` (str): ChatEngine plugin ID used as the inner LLM.
    - ``toolboxes`` (List[str]): ToolBox plugin IDs to load (inherited).
    - ``max_follow_ups`` (int): Maximum sub-questions before forcing a final
      answer (default: 8).

    Entry point group: ``opm.agents.chat``
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialise the Self-Ask engine.

        Args:
            config: Plugin configuration dict.
        """
        super().__init__(config=config)
        self._brain: Optional[ChatEngine] = None

    @property
    def max_follow_ups(self) -> int:
        """Maximum number of follow-up sub-questions before forcing a final answer."""
        return int(self.config.get("max_follow_ups", 8))

    @property
    def brain(self) -> Optional[ChatEngine]:
        """The inner ChatEngine used for all LLM calls."""
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

    def _get_tool_names(self) -> List[str]:
        """
        Collect names of all available tools across registered toolboxes.

        Returns:
            Sorted list of tool name strings.
        """
        names: List[str] = []
        for tb in self.toolboxes:
            try:
                for tool in tb.discover_tools():
                    names.append(tool.name)
            except Exception:  # noqa: BLE001
                pass
        return sorted(names)

    def _call_first_matching_tool(self, query: str) -> str:
        """
        Call the first available tool with a plain-text query.

        Iterates toolboxes in registration order.  Passes ``{"query": query}``
        as the argument; if that fails, falls back to ``{"q": query}`` and
        finally ``{"text": query}``.

        Args:
            query: Plain text search / lookup query.

        Returns:
            String representation of the tool output, or an error message.
        """
        for tb in self.toolboxes:
            try:
                tools = tb.discover_tools()
                if not tools:
                    continue
                tool_name = tools[0].name
                for key in ("query", "q", "text"):
                    try:
                        result = tb.call_tool(tool_name, {key: query})
                        return str(result)
                    except Exception:  # noqa: BLE001
                        continue
            except Exception:  # noqa: BLE001
                continue
        return f"(No tool available to answer: {query})"

    def _call_named_tool(self, tool_name: str, query: str) -> str:
        """
        Call a specifically-named tool with a plain-text query.

        Args:
            tool_name: Name of the tool to invoke.
            query: Plain text query string.

        Returns:
            String representation of the tool output, or an error message.
        """
        for tb in self.toolboxes:
            try:
                tool = tb.get_tool(tool_name)
                if tool is not None:
                    for key in ("query", "q", "text"):
                        try:
                            result = tb.call_tool(tool_name, {key: query})
                            return str(result)
                        except Exception:  # noqa: BLE001
                            continue
            except Exception:  # noqa: BLE001
                continue
        return f"Error: tool '{tool_name}' not found."

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        """
        Run the Self-Ask loop and return the final response.

        The loop appends each ``Follow up:`` / ``Intermediate answer:`` exchange
        to a running transcript that the brain uses to decide its next step.

        Args:
            messages: Conversation history including the latest user turn.
            session_id: Session identifier forwarded to the brain.
            lang: BCP-47 language code forwarded to the brain.
            units: Measurement system forwarded to the brain.

        Returns:
            ``AgentMessage`` with ``MessageRole.ASSISTANT`` containing the
            final synthesized answer.
        """
        if self.brain is None:
            return AgentMessage(role=MessageRole.ASSISTANT,
                                content="Error: no brain configured.")

        tool_names = self._get_tool_names()

        if tool_names:
            tool_note = _TOOL_NOTE_WITH_SEARCH.format(
                tool_names=", ".join(tool_names)
            )
        else:
            tool_note = _TOOL_NOTE_NO_TOOLS

        system_msg = AgentMessage(
            role=MessageRole.SYSTEM,
            content=_SELF_ASK_SYSTEM_PROMPT.format(tool_note=tool_note),
        )
        loop_messages: List[AgentMessage] = [system_msg, *messages]

        for _ in range(self.max_follow_ups):
            response = self.brain.continue_chat(
                loop_messages, session_id=session_id, lang=lang, units=units
            )
            text = response.content

            # Check for final answer.
            final = _extract_final_answer(text)
            if final is not None:
                return AgentMessage(role=MessageRole.ASSISTANT, content=final)

            # Check for an explicit tool call block.
            tool_call = _extract_tool_call(text)
            if tool_call:
                tool_name, query = tool_call
                observation = self._call_named_tool(tool_name, query)
                loop_messages.append(AgentMessage(role=MessageRole.ASSISTANT, content=text))
                loop_messages.append(AgentMessage(
                    role=MessageRole.USER,
                    content=f"{_INTERMEDIATE_PREFIX} {observation}",
                ))
                continue

            # Check for a follow-up question (no explicit tool call).
            follow_up = _extract_follow_up(text)
            if follow_up and tool_names:
                observation = self._call_first_matching_tool(follow_up)
                loop_messages.append(AgentMessage(role=MessageRole.ASSISTANT, content=text))
                loop_messages.append(AgentMessage(
                    role=MessageRole.USER,
                    content=f"{_INTERMEDIATE_PREFIX} {observation}",
                ))
                continue

            # No recognized pattern — treat as final answer.
            return AgentMessage(role=MessageRole.ASSISTANT, content=text)

        # Max follow-ups reached — force a final answer.
        loop_messages.append(AgentMessage(
            role=MessageRole.USER,
            content="You have asked the maximum number of follow-up questions. "
                    "Synthesize what you know and answer: "
                    f"{_FINAL_ANSWER_PREFIX}",
        ))
        response = self.brain.continue_chat(
            loop_messages, session_id=session_id, lang=lang, units=units
        )
        final = _extract_final_answer(response.content) or response.content
        return AgentMessage(role=MessageRole.ASSISTANT, content=final)
