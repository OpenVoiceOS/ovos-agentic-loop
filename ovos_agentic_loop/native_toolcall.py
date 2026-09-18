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

"""NativeToolCallEngine — agent loop driven by provider-native tool calls.

Unlike :class:`ReActLoopEngine`, which prompts the brain to emit ``Action:`` /
``Observation:`` text and regex-parses it, this engine hands the toolbox schemas
to the brain via the ``tools`` argument and reads back **structured**
``AgentMessage.tool_calls``. It executes each call through the existing
``ToolBox`` machinery and feeds the results back as ``MessageRole.TOOL`` messages
until the brain answers without requesting tools.

When the configured brain does not advertise ``supports_tools`` (i.e. has no
native function-calling), this engine transparently falls back to the inherited
ReAct text loop, so it works with any brain.
"""
from typing import Any, Dict, List, Optional

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

from ovos_agentic_loop.react import ReActLoopEngine


class NativeToolCallEngine(ReActLoopEngine):
    """Agentic loop using native ``tool_calls`` with a ReAct text fallback.

    Reuses ``ReActLoopEngine``'s brain wiring, toolbox loading, schema collection
    and tool execution; only the loop in :meth:`continue_chat` differs.

    Config keys (inherited): ``brain`` (ChatEngine plugin id), ``toolboxes``
    (List[str]), ``max_iterations`` (int, default 10).

    Entry point group: ``opm.agents.chat``
    """

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None,
                      tools: Optional[List[Dict[str, Any]]] = None) -> AgentMessage:
        """Run the native tool-calling loop and return the final assistant message.

        If the brain lacks ``supports_tools`` this delegates to the ReAct text
        loop (``super().continue_chat``). Otherwise it offers the toolbox schemas
        to the brain and, while the brain returns ``tool_calls``, executes each
        and appends a ``MessageRole.TOOL`` result (the assistant turn carrying the
        ``tool_calls`` is appended first, preserving the provider ordering
        invariant). On reaching ``max_iterations`` it makes one final tool-free
        call to force a text answer.

        Args:
            messages: Conversation history including the latest user turn.
            session_id / lang / units: Forwarded to the brain.
            tools: Ignored — schemas are collected from the registered toolboxes.

        Returns:
            ``AgentMessage`` with ``MessageRole.ASSISTANT`` containing the answer.
        """
        if self.brain is None:
            return AgentMessage(role=MessageRole.ASSISTANT,
                                content="Error: no brain configured.")

        # No native function-calling on this brain → ReAct text loop.
        if not getattr(self.brain, "supports_tools", False):
            return super().continue_chat(messages, session_id, lang, units)

        # Pass the ToolBox objects straight through; the brain normalizes them
        # (via ToolBox.normalize_tools) to its provider's tool format.
        tools = list(self.toolboxes)
        loop_messages: List[AgentMessage] = list(messages)

        for _ in range(self.max_iterations):
            response = self.brain.continue_chat(
                loop_messages, session_id=session_id, lang=lang, units=units,
                tools=tools,
            )
            if not response.tool_calls:
                return AgentMessage(role=MessageRole.ASSISTANT, content=response.content)

            # Assistant turn carrying tool_calls must precede its TOOL results.
            loop_messages.append(response)
            for tc in response.tool_calls:
                observation = self._call_tool(tc.name, tc.arguments)
                loop_messages.append(AgentMessage(
                    role=MessageRole.TOOL,
                    content=observation,
                    tool_call_id=tc.id,
                    name=tc.name,
                ))

        # Max iterations reached — one final, tool-free call to force an answer.
        final = self.brain.continue_chat(
            loop_messages, session_id=session_id, lang=lang, units=units, tools=None,
        )
        return AgentMessage(role=MessageRole.ASSISTANT, content=final.content)
