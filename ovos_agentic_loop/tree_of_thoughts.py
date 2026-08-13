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
"""TreeOfThoughtsEngine — beam search over reasoning branches.

Based on "Tree of Thoughts: Deliberate Problem Solving with Large Language
Models" (Yao et al., 2023 — https://arxiv.org/abs/2305.10601).

Algorithm
---------
At each depth level the engine:

1. **Expand**: generates ``n_branches`` independent candidate continuations
   (thoughts) from the current best branch.
2. **Evaluate**: scores each candidate on a 0–10 scale using a separate
   evaluator LLM call.
3. **Select**: keeps the top ``beam_width`` candidates (beam search).
4. **Terminate**: if any candidate contains a final answer marker, or
   ``max_depth`` is reached, the best-scored branch is returned.

This implements **BFS with beam pruning** as described in the original paper.
DFS with backtracking (the paper's other variant) is not implemented here
because it requires state rollback that is expensive with LLM context windows.

Key differences from ReAct
--------------------------
- Explores **multiple parallel reasoning paths** simultaneously.
- A separate **evaluator** judges which paths are worth continuing.
- Can discover non-obvious solution strategies by keeping 2–3 competing
  hypotheses alive across multiple steps.
- Much more expensive: ``n_branches × (1 generator + 1 evaluator)`` calls
  per depth level.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ovos_plugin_manager.templates.agents import AgentMessage, ChatEngine, MessageRole, ToolsArg

from ovos_agentic_loop.base import AgenticLoopEngine

_THOUGHT_GENERATOR_PROMPT = """\
You are solving a problem step by step.

Problem: {problem}

{history}
Generate the next reasoning step.  Think creatively — this is ONE of several
parallel approaches being explored.  Be concise (2–4 sentences).
Do NOT write "Step N:" — just write the reasoning.
If you have reached a complete answer, end with: ANSWER: <your answer>
"""

_EVALUATOR_PROMPT = """\
Rate the following reasoning step for solving the given problem.

Problem: {problem}

Reasoning so far:
{history}

New step: {thought}

Score this step 0–10 where:
  10 = clearly correct, directly advances to the solution
   5 = plausible but uncertain
   0 = clearly wrong or irrelevant

Reply with ONLY a number (0–10) on the first line, then one sentence explaining.
"""

_ANSWER_MARKER = "ANSWER:"


@dataclass
class _Branch:
    """A single reasoning branch in the search tree."""

    history: List[str] = field(default_factory=list)
    score: float = 0.0
    final_answer: Optional[str] = None

    @property
    def history_text(self) -> str:
        """Format the accumulated thought history as a numbered list."""
        if not self.history:
            return "(no steps yet)"
        return "\n".join(f"Step {i + 1}: {t}" for i, t in enumerate(self.history))


def _extract_answer(text: str) -> Optional[str]:
    """
    Extract the text following ``ANSWER:`` in a thought.

    Args:
        text: A single generated thought.

    Returns:
        The answer string, or ``None`` if the marker is absent.
    """
    idx = text.upper().find(_ANSWER_MARKER.upper())
    if idx == -1:
        return None
    return text[idx + len(_ANSWER_MARKER):].strip()


def _parse_score(text: str) -> float:
    """
    Extract the numeric score from an evaluator response.

    Args:
        text: Raw evaluator LLM output.

    Returns:
        Float score 0.0–10.0; defaults to 5.0 if no number is found.
    """
    match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
    if match:
        return min(10.0, max(0.0, float(match.group(1))))
    return 5.0


class TreeOfThoughtsEngine(AgenticLoopEngine):
    """
    ``AgenticLoopEngine`` implementing Beam-Search Tree of Thoughts.

    At each depth level ``n_branches`` candidate reasoning steps are
    generated independently, evaluated for quality, and the top
    ``beam_width`` are carried forward.  Terminates when any branch
    produces an ``ANSWER:`` or ``max_depth`` levels are exhausted.

    Config keys:

    - ``brain`` (str): ChatEngine plugin ID used as inner LLM.
    - ``n_branches`` (int): Candidate thoughts to generate per step (default: 3).
    - ``beam_width`` (int): Branches to keep after evaluation (default: 2).
    - ``max_depth`` (int): Maximum reasoning depth before forcing an answer (default: 4).

    Entry point group: ``opm.agents.chat``
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialise the Tree-of-Thoughts engine.

        Args:
            config: Plugin configuration dict.
        """
        super().__init__(config=config)
        self._brain: Optional[ChatEngine] = None

    @property
    def n_branches(self) -> int:
        """Number of candidate thought branches generated per depth level."""
        return int(self.config.get("n_branches", 3))

    @property
    def beam_width(self) -> int:
        """Number of top branches to keep after evaluation at each level."""
        return int(self.config.get("beam_width", 2))

    @property
    def max_depth(self) -> int:
        """Maximum reasoning depth before forcing the best branch to answer."""
        return int(self.config.get("max_depth", 4))

    @property
    def brain(self) -> Optional[ChatEngine]:
        """The inner ChatEngine used for generation and evaluation."""
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

    def _generate_thought(self, problem: str, branch: _Branch,
                          session_id: str, lang: Optional[str],
                          units: Optional[str]) -> str:
        """
        Generate one candidate reasoning step continuing a branch.

        Args:
            problem: The original user problem statement.
            branch: The branch to continue.
            session_id: Session identifier.
            lang: BCP-47 language code.
            units: Measurement system.

        Returns:
            Raw thought string from the LLM.
        """
        prompt = _THOUGHT_GENERATOR_PROMPT.format(
            problem=problem,
            history=branch.history_text,
        )
        response = self.brain.continue_chat(
            [AgentMessage(role=MessageRole.USER, content=prompt)],
            session_id=session_id, lang=lang, units=units,
        )
        return response.content.strip()

    def _evaluate_thought(self, problem: str, branch: _Branch, thought: str,
                          session_id: str, lang: Optional[str],
                          units: Optional[str]) -> float:
        """
        Score a candidate thought continuation.

        Args:
            problem: The original user problem statement.
            branch: The parent branch.
            thought: The candidate thought to score.
            session_id: Session identifier.
            lang: BCP-47 language code.
            units: Measurement system.

        Returns:
            Score 0.0–10.0.
        """
        prompt = _EVALUATOR_PROMPT.format(
            problem=problem,
            history=branch.history_text,
            thought=thought,
        )
        response = self.brain.continue_chat(
            [AgentMessage(role=MessageRole.USER, content=prompt)],
            session_id=session_id, lang=lang, units=units,
        )
        return _parse_score(response.content)

    def _force_answer(self, problem: str, branch: _Branch,
                      session_id: str, lang: Optional[str],
                      units: Optional[str]) -> str:
        """
        Ask the LLM to produce a final answer from the best branch's history.

        Called when ``max_depth`` is reached without a natural ``ANSWER:``.

        Args:
            problem: The original user problem statement.
            branch: The highest-scoring surviving branch.
            session_id: Session identifier.
            lang: BCP-47 language code.
            units: Measurement system.

        Returns:
            Final answer string.
        """
        prompt = (
            f"Problem: {problem}\n\n"
            f"Your reasoning so far:\n{branch.history_text}\n\n"
            f"Based on this reasoning, what is the final answer?\n"
            f"Reply with: {_ANSWER_MARKER} <answer>"
        )
        response = self.brain.continue_chat(
            [AgentMessage(role=MessageRole.USER, content=prompt)],
            session_id=session_id, lang=lang, units=units,
        )
        return _extract_answer(response.content) or response.content

    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None,
                      tools: "ToolsArg" = None) -> AgentMessage:
        """
        Run the Tree-of-Thoughts beam search and return the best answer.

        At each depth level generates ``n_branches`` thoughts from each
        surviving branch, scores all candidates, and keeps the top
        ``beam_width`` for the next level.

        Args:
            messages: Conversation history including the latest user turn.
            session_id: Session identifier forwarded to the brain.
            lang: BCP-47 language code forwarded to the brain.
            units: Measurement system forwarded to the brain.

        Returns:
            ``AgentMessage`` with ``MessageRole.ASSISTANT`` containing the
            answer from the highest-scoring completed branch.
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

        # Extract the problem statement from the latest user message.
        problem = ""
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                problem = msg.content
                break

        # Initialise with a single empty root branch.
        live_branches: List[_Branch] = [_Branch()]

        for depth in range(self.max_depth):
            candidates: List[Tuple[_Branch, str, float]] = []

            for branch in live_branches:
                for _ in range(self.n_branches):
                    thought = self._generate_thought(
                        problem, branch, session_id, lang, units
                    )
                    answer = _extract_answer(thought)
                    if answer:
                        # Branch completed — return immediately (best first).
                        return AgentMessage(
                            role=MessageRole.ASSISTANT, content=answer
                        )
                    score = self._evaluate_thought(
                        problem, branch, thought, session_id, lang, units
                    )
                    candidates.append((branch, thought, score))

            # Sort by score descending, keep top beam_width.
            candidates.sort(key=lambda x: x[2], reverse=True)
            new_branches: List[_Branch] = []
            for branch, thought, score in candidates[: self.beam_width]:
                new_branch = _Branch(
                    history=branch.history + [thought],
                    score=score,
                )
                new_branches.append(new_branch)

            live_branches = new_branches

        # Max depth reached — force an answer from the best branch.
        best = max(live_branches, key=lambda b: b.score)
        answer = self._force_answer(problem, best, session_id, lang, units)
        return AgentMessage(role=MessageRole.ASSISTANT, content=answer)
