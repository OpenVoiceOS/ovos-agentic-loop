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

"""ReActLoopEngine — ReAct (Reason + Act) agentic loop implementation."""
import json
import re
from typing import Any, Dict, List, Optional

from ovos_plugin_manager.templates.agents import AgentMessage, ChatEngine, MessageRole, ToolsArg
from ovos_utils.log import LOG

from ovos_agentic_loop.base import AgenticLoopEngine

# Sentinel that tells the loop the agent is done iterating.
_FINAL_ANSWER_TOKEN = "FINAL_ANSWER:"

# System prompt injected ahead of the user-defined system prompt when tools
# are available.  The LLM must follow this structured output format.
_REACT_SYSTEM_PROMPT = """You have access to tools. On each turn you MUST choose one of:

1. Use a tool:
   Thought: <reason about what to do>
   Action: <tool_name>
   Action Input: <JSON object matching the tool's argument schema>

2. Give the final answer (only when you have enough information):
   Thought: <reason>
   {final_answer_token} <your answer to the user>

Available tools (JSON schema):
{{tool_schemas}}

Rules:
- Never skip the Thought line.
- Action Input MUST be valid JSON.
- Call only ONE tool per turn.
- After receiving an Observation, continue reasoning.
- Use {final_answer_token} only once, as the last step.
""".format(final_answer_token=_FINAL_ANSWER_TOKEN)


def _build_react_system(tool_schemas: List[Dict[str, Any]]) -> str:
    """
    Build the ReAct system prompt with tool schemas embedded.

    Args:
        tool_schemas: List of JSON-schema dicts as returned by
            ``ToolBox.tool_json_list``.

    Returns:
        Formatted system prompt string.
    """
    return _REACT_SYSTEM_PROMPT.replace(
        "{tool_schemas}", json.dumps(tool_schemas, indent=2)
    )


def _extract_json_object(text: str, start: int) -> Optional[str]:
    """
    Extract a complete, balanced JSON object starting at ``text[start]``.

    Walks the string counting ``{`` / ``}`` to find the true closing brace,
    correctly skipping braces inside string literals and handling escaped
    characters.  This is necessary because the non-greedy regex ``\\{.*?\\}``
    stops at the first ``}`` and truncates nested JSON objects.

    Args:
        text: Source string.
        start: Index of the opening ``{``.

    Returns:
        The full JSON substring, or ``None`` if the object is unterminated.
    """
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_action(text: str) -> Optional[tuple]:
    """
    Extract ``(tool_name, tool_input_dict)`` from a ReAct-formatted response.

    Uses a balanced-brace parser for the JSON argument object so that nested
    dicts (e.g. ``{"filter": {"date": "today"}}``) are captured correctly.

    Args:
        text: Raw LLM output text.

    Returns:
        ``(tool_name, args_dict)`` or ``None`` if no action was found.
    """
    action_match = re.search(r"Action:\s*(\S+)", text)
    # Find the start of the JSON object following "Action Input:"
    input_start_match = re.search(r"Action Input:\s*(\{)", text)
    if not action_match or not input_start_match:
        return None
    tool_name = action_match.group(1).strip()
    json_start = input_start_match.start(1)
    json_str = _extract_json_object(text, json_start)
    if json_str is None:
        return None
    try:
        args = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    return tool_name, args


def _extract_final_answer(text: str) -> Optional[str]:
    """
    Extract the answer following ``FINAL_ANSWER:`` in the LLM output.

    Args:
        text: Raw LLM output text.

    Returns:
        The answer string, or ``None`` if the token was not found.
    """
    idx = text.find(_FINAL_ANSWER_TOKEN)
    if idx == -1:
        return None
    return text[idx + len(_FINAL_ANSWER_TOKEN):].strip()


class ReActLoopEngine(AgenticLoopEngine):
    """
    Concrete ``AgenticLoopEngine`` implementing the ReAct pattern.

    Each iteration the brain LLM generates a *Thought*, an optional *Action*
    (tool call), receives an *Observation* (tool result), then repeats.
    Iteration stops when the LLM emits ``FINAL_ANSWER:`` or
    ``max_iterations`` is reached.

    Config keys:

    - ``brain`` (str): ChatEngine plugin ID used as the inner LLM.
    - ``toolboxes`` (List[str]): ToolBox plugin IDs to load (inherited).
    - ``max_iterations`` (int): Maximum tool-call cycles (default: 10).

    Entry point group: ``opm.agents.chat``
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialise the ReAct engine.

        Args:
            config: Plugin configuration.  Must contain a ``"brain"`` key
                naming a ChatEngine plugin ID unless ``set_brain()`` is called
                before the first ``continue_chat`` invocation.
        """
        super().__init__(config=config)
        self._brain: Optional[ChatEngine] = None

    @property
    def max_iterations(self) -> int:
        """Maximum number of tool-call cycles before forcing a final answer."""
        return int(self.config.get("max_iterations", 10))

    @property
    def brain(self) -> Optional[ChatEngine]:
        """The inner ChatEngine used for LLM calls."""
        if self._brain is None:
            self._brain = self._load_brain()
            if self._brain is not None:
                self._inject_brain_into_toolboxes(self._brain)
        return self._brain

    def set_brain(self, brain: ChatEngine) -> None:
        """
        Inject a ChatEngine instance as the inner LLM.

        Also propagates the brain to any registered toolboxes that expose a
        ``set_brain`` method (e.g. ``SkillMDToolBox``).

        Args:
            brain: Instantiated ``ChatEngine`` to use for all LLM calls.
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
        except Exception as exc:  # noqa: BLE001
            LOG.warning(f"ReActLoopEngine: failed to load brain '{brain_id}': {exc}")
            return None

    def _collect_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Gather JSON schemas from all registered toolboxes.

        Returns:
            Flat list of tool schema dicts suitable for embedding in the system
            prompt.
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

        Searches toolboxes in registration order, calls the first match.

        Args:
            tool_name: Name of the tool to invoke.
            args: Keyword arguments to pass to the tool.

        Returns:
            String representation of the tool output.
        """
        for tb in self.toolboxes:
            try:
                tool = tb.get_tool(tool_name)
            except Exception as e:  # noqa: BLE001 - lookup failed; try the next toolbox
                LOG.debug(f"toolbox {tb} lookup for '{tool_name}' failed: {e}")
                continue
            if tool is None:
                continue
            try:
                result = tb.call_tool(tool_name, args)
                return str(result)
            except Exception as e:  # noqa: BLE001 - tool found but raised
                # Surface the real error as the observation so the loop can recover
                # (retry with different args / pick another tool) instead of being
                # told the tool was "not found".
                LOG.error(f"tool '{tool_name}' raised: {e}")
                return f"Error: tool '{tool_name}' failed: {e}"
        return f"Error: tool '{tool_name}' not found."

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None,
                      tools: "ToolsArg" = None) -> AgentMessage:
        """
        Run the ReAct loop and return the final assistant response.

        Prepends a ReAct system prompt (if tools are available) to the message
        list, then iterates: call brain → parse action → call tool → append
        observation, until a final answer is produced or ``max_iterations`` is
        reached.

        Args:
            messages: Conversation history including the latest user turn.
            session_id: Session identifier forwarded to the brain.
            lang: BCP-47 language code forwarded to the brain.
            units: Measurement system forwarded to the brain.

        Returns:
            ``AgentMessage`` with ``MessageRole.ASSISTANT`` containing the
            final answer.
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

        tool_schemas = self._collect_tool_schemas()
        loop_messages: List[AgentMessage] = list(messages)

        # Prepend ReAct instructions when tools are available.
        if tool_schemas:
            react_sys = AgentMessage(
                role=MessageRole.SYSTEM,
                content=_build_react_system(tool_schemas),
            )
            loop_messages = [react_sys] + loop_messages

        for _ in range(self.max_iterations):
            response = self.brain.continue_chat(
                loop_messages, session_id=session_id, lang=lang, units=units
            )
            text = response.content

            # Check for final answer first.
            final = _extract_final_answer(text)
            if final is not None:
                return AgentMessage(role=MessageRole.ASSISTANT, content=final)

            # Try to parse and execute a tool action.
            parsed = _parse_action(text)
            if parsed is None:
                # No action found — treat entire response as final answer.
                return AgentMessage(role=MessageRole.ASSISTANT, content=text)

            tool_name, args = parsed
            observation = self._call_tool(tool_name, args)

            # Append the assistant's reasoning step and the observation.
            loop_messages.append(AgentMessage(role=MessageRole.ASSISTANT, content=text))
            loop_messages.append(AgentMessage(
                role=MessageRole.USER,
                content=f"Observation: {observation}",
            ))

        # Max iterations reached — ask brain for a final answer.
        loop_messages.append(AgentMessage(
            role=MessageRole.USER,
            content=f"You have used the maximum number of tool calls. "
                    f"Provide your best {_FINAL_ANSWER_TOKEN} now.",
        ))
        response = self.brain.continue_chat(
            loop_messages, session_id=session_id, lang=lang, units=units
        )
        final = _extract_final_answer(response.content) or response.content
        return AgentMessage(role=MessageRole.ASSISTANT, content=final)
