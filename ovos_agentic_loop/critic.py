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
"""CRITICEngine — answer first, then verify with tools, then revise.

Based on "CRITIC: Large Language Models Can Self-Correct with Tool-Interactive
Critiquing" (Gou et al., 2023 — https://arxiv.org/abs/2305.11738).

Algorithm
---------
1. **Draft**: the brain generates an initial answer to the user's question.
2. **Critique**: the brain identifies specific claims in the draft that should be
   verified externally (e.g. facts, dates, values).  For each claim it produces
   a *critique* and a *tool call* to check it.
3. **Verify**: each tool call is executed; the observation is appended.
4. **Revise**: after all critiques are resolved the brain produces a revised,
   corrected final answer.

The loop repeats for up to ``max_critique_rounds`` rounds.  If the brain
reports ``VERIFIED: all claims are correct`` the draft is returned unchanged.

Key differences from Reflexion
-------------------------------
- Verification is **tool-assisted** — the agent checks facts externally, not
  by evaluating its own reasoning.
- The brain critiques a *specific answer* rather than a *process* — better
  suited to factual tasks (questions with verifiable ground truth).
- Reflexion catches *reasoning errors*; CRITIC catches *factual errors*.
"""
import re
from typing import Any, Dict, List, Optional

from ovos_plugin_manager.templates.agents import AgentMessage, ChatEngine, MessageRole, ToolsArg

from ovos_agentic_loop.base import AgenticLoopEngine

_DRAFT_SYSTEM_PROMPT = """\
Answer the user's question as clearly and accurately as you can.
"""

_CRITIQUE_SYSTEM_PROMPT = """\
You are a fact-checking assistant.  You will be given a question and a draft answer.

Your job:
1. Identify every factual claim in the draft that can be verified with a tool.
2. For each claim, emit one critique block:

   CLAIM: <the specific claim to verify>
   TOOL: <tool_name>
   TOOL INPUT: <plain text query>

3. If there are NO claims to verify (the answer is purely subjective or
   already verified), emit exactly:
   VERIFIED: all claims are correct

Rules:
- Emit only CLAIM/TOOL/TOOL INPUT blocks or a single VERIFIED line.
- One block per claim — do not bundle multiple claims.
- Available tools: {tool_names}
"""

_REVISE_SYSTEM_PROMPT = """\
You are revising a draft answer based on tool verification results.

Original question: {question}

Draft answer: {draft}

Verification results:
{verifications}

Rewrite the answer incorporating the verified information.
Correct any errors found.  If all facts checked out, you may return the
draft unchanged.
Reply with ONLY the revised answer — no preamble.
"""

_VERIFIED_SENTINEL = "VERIFIED:"


def _parse_critique_blocks(text: str) -> List[Dict[str, str]]:
    """
    Extract ``CLAIM / TOOL / TOOL INPUT`` blocks from a critique response.

    Args:
        text: Raw LLM critique output.

    Returns:
        List of dicts with keys ``claim``, ``tool``, ``tool_input``.
    """
    blocks: List[Dict[str, str]] = []
    # Split on CLAIM: to get individual blocks.
    parts = re.split(r"(?i)CLAIM:\s*", text)
    for part in parts[1:]:  # skip text before first CLAIM
        claim_match = re.match(r"(.+?)(?=TOOL:)", part, re.DOTALL | re.IGNORECASE)
        tool_match = re.search(r"(?i)TOOL:\s*(\S+)", part)
        input_match = re.search(r"(?i)TOOL INPUT:\s*(.+?)(?=CLAIM:|$)", part, re.DOTALL)
        if claim_match and tool_match and input_match:
            blocks.append({
                "claim": claim_match.group(1).strip(),
                "tool": tool_match.group(1).strip(),
                "tool_input": input_match.group(1).strip(),
            })
    return blocks


class CRITICEngine(AgenticLoopEngine):
    """
    ``AgenticLoopEngine`` implementing the CRITIC pattern.

    The brain first drafts an answer, then iteratively critiques its own
    claims by calling tools to verify them.  The verified observations are
    used to revise the answer up to ``max_critique_rounds`` times.

    Best for **factual questions** where the answer contains verifiable
    claims and a search/lookup tool is available.

    Config keys:

    - ``brain`` (str): ChatEngine plugin ID used as inner LLM.
    - ``toolboxes`` (List[str]): ToolBox plugin IDs to load (inherited).
    - ``max_critique_rounds`` (int): Maximum critique → verify → revise
      cycles (default: 2).

    Entry point group: ``opm.agents.chat``
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialise the CRITIC engine.

        Args:
            config: Plugin configuration dict.
        """
        super().__init__(config=config)
        self._brain: Optional[ChatEngine] = None

    @property
    def max_critique_rounds(self) -> int:
        """Maximum critique → verify → revise cycles."""
        return int(self.config.get("max_critique_rounds", 2))

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

    def _call_tool(self, tool_name: str, query: str) -> str:
        """
        Call a named tool with a plain-text query.

        Tries ``query``, ``q``, and ``text`` as argument key names in order.

        Args:
            tool_name: Name of the tool to invoke.
            query: Plain text query string.

        Returns:
            String representation of the tool output, or an error message.
        """
        for tb in self.toolboxes:
            try:
                if tb.get_tool(tool_name) is not None:
                    for key in ("query", "q", "text"):
                        try:
                            return str(tb.call_tool(tool_name, {key: query}))
                        except Exception:  # noqa: BLE001
                            continue
            except Exception:  # noqa: BLE001
                continue
        return f"Error: tool '{tool_name}' not found or failed."

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None,
                      tools: "ToolsArg" = None) -> AgentMessage:
        """
        Run the CRITIC loop and return the verified, revised answer.

        Phases: (1) draft answer, (2) critique → tool verify, (3) revise.
        Repeats phases 2–3 up to ``max_critique_rounds`` times.

        Args:
            messages: Conversation history including the latest user turn.
            session_id: Session identifier forwarded to the brain.
            lang: BCP-47 language code forwarded to the brain.
            units: Measurement system forwarded to the brain.

        Returns:
            ``AgentMessage`` with ``MessageRole.ASSISTANT`` containing the
            verified and revised answer.
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

        tool_names = self._get_tool_names()

        # Extract original question for the revise prompt.
        question = ""
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                question = msg.content
                break

        # --- Phase 1: Draft ---
        draft_messages = [
            AgentMessage(role=MessageRole.SYSTEM, content=_DRAFT_SYSTEM_PROMPT),
            *messages,
        ]
        draft_response = self.brain.continue_chat(
            draft_messages, session_id=session_id, lang=lang, units=units
        )
        current_answer = draft_response.content

        # If no tools, skip critique phases.
        if not tool_names:
            return AgentMessage(role=MessageRole.ASSISTANT, content=current_answer)

        # --- Phases 2–3: Critique → Verify → Revise (repeated) ---
        critique_system = _CRITIQUE_SYSTEM_PROMPT.format(
            tool_names=", ".join(tool_names)
        )

        for _ in range(self.max_critique_rounds):
            critique_messages = [
                AgentMessage(role=MessageRole.SYSTEM, content=critique_system),
                AgentMessage(
                    role=MessageRole.USER,
                    content=f"Question: {question}\n\nDraft answer: {current_answer}",
                ),
            ]
            critique_response = self.brain.continue_chat(
                critique_messages, session_id=session_id, lang=lang, units=units
            )
            critique_text = critique_response.content

            # If the brain says all is verified, we're done.
            if _VERIFIED_SENTINEL.lower() in critique_text.lower():
                break

            blocks = _parse_critique_blocks(critique_text)
            if not blocks:
                break  # Nothing to verify.

            # Verify each claim via tool calls.
            verifications: List[str] = []
            for block in blocks:
                observation = self._call_tool(block["tool"], block["tool_input"])
                verifications.append(
                    f'Claim: "{block["claim"]}"\n'
                    f'Tool result: {observation}'
                )

            # Revise the answer with the verification results.
            revise_prompt = _REVISE_SYSTEM_PROMPT.format(
                question=question,
                draft=current_answer,
                verifications="\n\n".join(verifications),
            )
            revise_response = self.brain.continue_chat(
                [AgentMessage(role=MessageRole.USER, content=revise_prompt)],
                session_id=session_id, lang=lang, units=units,
            )
            current_answer = revise_response.content

        return AgentMessage(role=MessageRole.ASSISTANT, content=current_answer)
