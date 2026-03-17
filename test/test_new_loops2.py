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
"""Tests for ChainOfThoughtEngine, CRITICEngine, TreeOfThoughtsEngine."""
import unittest
from typing import List
from unittest.mock import MagicMock

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

from ovos_agentic_loop.chain_of_thought import (
    ChainOfThoughtEngine,
    _extract_final_answer as cot_extract,
)
from ovos_agentic_loop.critic import (
    CRITICEngine,
    _parse_critique_blocks,
)
from ovos_agentic_loop.tree_of_thoughts import (
    TreeOfThoughtsEngine,
    _Branch,
    _extract_answer as tot_extract,
    _parse_score as tot_parse_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_brain(responses: List[str]) -> MagicMock:
    brain = MagicMock()
    brain.continue_chat.side_effect = [
        AgentMessage(role=MessageRole.ASSISTANT, content=r)
        for r in responses
    ]
    return brain


def _user(text: str) -> AgentMessage:
    return AgentMessage(role=MessageRole.USER, content=text)


# ===========================================================================
# ChainOfThoughtEngine
# ===========================================================================

class TestCotExtract(unittest.TestCase):
    def test_extracts_final_answer(self):
        text = "Step 1: 2+2=4\nStep 2: 4+4=8\nFINAL ANSWER: 8"
        self.assertEqual(cot_extract(text), "8")

    def test_case_insensitive(self):
        self.assertEqual(cot_extract("final answer: yes"), "yes")

    def test_missing_marker(self):
        self.assertIsNone(cot_extract("No marker here."))


class TestChainOfThoughtEngine(unittest.TestCase):
    def test_no_brain_error(self):
        engine = ChainOfThoughtEngine(config={})
        result = engine.continue_chat([_user("hello")])
        self.assertIn("Error", result.content)

    def test_extracts_final_answer(self):
        engine = ChainOfThoughtEngine(config={})
        engine.set_brain(_make_brain(["Step 1: Think.\nFINAL ANSWER: 42"]))
        result = engine.continue_chat([_user("What is 6*7?")])
        self.assertEqual(result.content, "42")
        self.assertEqual(result.role, MessageRole.ASSISTANT)

    def test_returns_full_response_when_no_marker(self):
        engine = ChainOfThoughtEngine(config={})
        engine.set_brain(_make_brain(["I think the answer is 42."]))
        result = engine.continue_chat([_user("What is 6*7?")])
        self.assertEqual(result.content, "I think the answer is 42.")

    def test_single_llm_call(self):
        """CoT must not iterate — exactly one brain call per continue_chat."""
        brain = _make_brain(["FINAL ANSWER: done"])
        engine = ChainOfThoughtEngine(config={})
        engine.set_brain(brain)
        engine.continue_chat([_user("question")])
        self.assertEqual(brain.continue_chat.call_count, 1)

    def test_system_prompt_config_prepended(self):
        """extra system_prompt from config is prepended before CoT instructions."""
        brain = _make_brain(["FINAL ANSWER: ok"])
        engine = ChainOfThoughtEngine(config={"system_prompt": "You are a helpful bot."})
        engine.set_brain(brain)
        engine.continue_chat([_user("hi")])
        call_args = brain.continue_chat.call_args[0][0]
        system_msg = call_args[0]
        self.assertIn("You are a helpful bot.", system_msg.content)
        self.assertIn("step by step", system_msg.content)

    def test_ignores_toolboxes(self):
        """Registered toolboxes are not called in CoT."""
        tb = MagicMock()
        engine = ChainOfThoughtEngine(config={})
        engine.set_brain(_make_brain(["FINAL ANSWER: x"]))
        engine.load_toolboxes([tb])
        engine.continue_chat([_user("q")])
        tb.call_tool.assert_not_called()


# ===========================================================================
# CRITICEngine helpers
# ===========================================================================

class TestParseCritiqueBlocks(unittest.TestCase):
    def test_single_block(self):
        text = (
            "CLAIM: The Eiffel Tower is in Berlin.\n"
            "TOOL: web_search\n"
            "TOOL INPUT: Eiffel Tower location\n"
        )
        blocks = _parse_critique_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertIn("Eiffel Tower", blocks[0]["claim"])
        self.assertEqual(blocks[0]["tool"], "web_search")
        self.assertEqual(blocks[0]["tool_input"], "Eiffel Tower location")

    def test_multiple_blocks(self):
        text = (
            "CLAIM: Claim A\nTOOL: search\nTOOL INPUT: query A\n"
            "CLAIM: Claim B\nTOOL: lookup\nTOOL INPUT: query B\n"
        )
        blocks = _parse_critique_blocks(text)
        self.assertEqual(len(blocks), 2)

    def test_empty(self):
        self.assertEqual(_parse_critique_blocks("VERIFIED: all correct"), [])

    def test_case_insensitive(self):
        text = "claim: X\ntool: search\ntool input: q\n"
        self.assertEqual(len(_parse_critique_blocks(text)), 1)




# ===========================================================================
# CRITICEngine integration
# ===========================================================================

class TestCRITICEngine(unittest.TestCase):
    def test_no_brain_error(self):
        engine = CRITICEngine(config={})
        result = engine.continue_chat([_user("hello")])
        self.assertIn("Error", result.content)

    def test_no_tools_returns_draft(self):
        """Without tools the draft is returned unchanged after phase 1."""
        engine = CRITICEngine(config={})
        engine.set_brain(_make_brain(["Paris is the capital of Germany."]))
        result = engine.continue_chat([_user("What is the capital of France?")])
        self.assertEqual(result.content, "Paris is the capital of Germany.")
        # Only 1 brain call (draft only).
        self.assertEqual(engine.brain.continue_chat.call_count, 1)

    def test_verified_stops_loop(self):
        """VERIFIED sentinel ends the critique loop without revising."""
        tb = MagicMock()
        tool = MagicMock()
        tool.name = "search"
        tb.discover_tools.return_value = [tool]
        engine = CRITICEngine(config={"max_critique_rounds": 3})
        engine.set_brain(_make_brain([
            "The Eiffel Tower is in Paris.",          # draft
            "VERIFIED: all claims are correct",       # critique → stop
        ]))
        engine.toolboxes = [tb]
        result = engine.continue_chat([_user("Where is the Eiffel Tower?")])
        self.assertEqual(result.content, "The Eiffel Tower is in Paris.")
        tb.call_tool.assert_not_called()

    def test_critique_calls_tool_and_revises(self):
        """One critique block triggers a tool call and revision."""
        tb = MagicMock()
        search_tool = MagicMock()
        search_tool.name = "web_search"
        tb.discover_tools.return_value = [search_tool]
        tb.get_tool.return_value = search_tool
        tb.call_tool.return_value = "Paris is in France."

        engine = CRITICEngine(config={"max_critique_rounds": 1})
        engine.set_brain(_make_brain([
            "The Eiffel Tower is in Berlin.",                            # draft
            "CLAIM: The Eiffel Tower is in Berlin.\n"                   # critique
            "TOOL: web_search\n"
            "TOOL INPUT: Eiffel Tower location\n",
            "The Eiffel Tower is in Paris, France.",                     # revised
        ]))
        engine.toolboxes = [tb]
        result = engine.continue_chat([_user("Where is the Eiffel Tower?")])
        self.assertEqual(result.content, "The Eiffel Tower is in Paris, France.")
        tb.call_tool.assert_called_once()

    def test_max_critique_rounds_respected(self):
        """Engine stops after max_critique_rounds even if more rounds could happen."""
        tb = MagicMock()
        t = MagicMock()
        t.name = "s"
        tb.discover_tools.return_value = [t]
        tb.get_tool.return_value = t
        tb.call_tool.return_value = "result"

        engine = CRITICEngine(config={"max_critique_rounds": 2})
        engine.set_brain(_make_brain([
            "Draft.",                                       # draft
            "CLAIM: X\nTOOL: s\nTOOL INPUT: q\n",          # critique round 1
            "Revised 1.",                                   # revise round 1
            "CLAIM: Y\nTOOL: s\nTOOL INPUT: q2\n",         # critique round 2
            "Revised 2.",                                   # revise round 2
        ]))
        engine.toolboxes = [tb]
        result = engine.continue_chat([_user("q")])
        self.assertEqual(result.content, "Revised 2.")
        self.assertEqual(tb.call_tool.call_count, 2)


# ===========================================================================
# TreeOfThoughtsEngine helpers
# ===========================================================================

class TestTotHelpers(unittest.TestCase):
    def test_extract_answer(self):
        self.assertEqual(tot_extract("ANSWER: Paris"), "Paris")

    def test_extract_answer_missing(self):
        self.assertIsNone(tot_extract("No answer yet."))

    def test_parse_score_integer(self):
        self.assertEqual(tot_parse_score("9\nGreat step."), 9.0)

    def test_parse_score_default(self):
        self.assertEqual(tot_parse_score("looks fine"), 5.0)

    def test_branch_history_text(self):
        b = _Branch(history=["Think A.", "Think B."])
        text = b.history_text
        self.assertIn("Step 1", text)
        self.assertIn("Think A.", text)
        self.assertIn("Step 2", text)

    def test_branch_empty_history(self):
        b = _Branch()
        self.assertIn("no steps", b.history_text)


# ===========================================================================
# TreeOfThoughtsEngine integration
# ===========================================================================

class TestTreeOfThoughtsEngine(unittest.TestCase):
    def test_no_brain_error(self):
        engine = TreeOfThoughtsEngine(config={})
        result = engine.continue_chat([_user("hello")])
        self.assertIn("Error", result.content)

    def test_answer_on_first_thought(self):
        """If the first generated thought contains ANSWER:, return immediately."""
        engine = TreeOfThoughtsEngine(config={"n_branches": 1, "beam_width": 1, "max_depth": 3})
        engine.set_brain(_make_brain(["ANSWER: Paris is the capital of France."]))
        result = engine.continue_chat([_user("Capital of France?")])
        self.assertEqual(result.content, "Paris is the capital of France.")
        # Only 1 brain call needed (generator, no evaluator).
        self.assertEqual(engine.brain.continue_chat.call_count, 1)

    def test_beam_search_selects_best(self):
        """With 2 branches and beam_width=1, the higher-scored branch survives."""
        # depth=1, n_branches=2: generate 2 thoughts, evaluate 2, keep 1, force answer.
        engine = TreeOfThoughtsEngine(config={"n_branches": 2, "beam_width": 1, "max_depth": 1})
        engine.set_brain(_make_brain([
            "Thought A (weak)",   # branch 0, thought 0
            "3\nWeak.",           # score for thought A
            "Thought B (strong)", # branch 0, thought 1
            "9\nStrong.",         # score for thought B
            "ANSWER: B wins.",    # force_answer
        ]))
        result = engine.continue_chat([_user("Which thought is better?")])
        self.assertEqual(result.content, "B wins.")

    def test_force_answer_at_max_depth(self):
        """When max_depth is reached without ANSWER:, force_answer is called."""
        engine = TreeOfThoughtsEngine(config={"n_branches": 1, "beam_width": 1, "max_depth": 2})
        engine.set_brain(_make_brain([
            "Step 1 thought.",      # depth 0 generation
            "7\nGood.",             # depth 0 evaluation
            "Step 2 thought.",      # depth 1 generation
            "7\nGood.",             # depth 1 evaluation
            "ANSWER: Final answer from forced step.",  # force_answer
        ]))
        result = engine.continue_chat([_user("Solve it.")])
        self.assertEqual(result.content, "Final answer from forced step.")

    def test_answer_extracted_from_force_answer_without_marker(self):
        """If force_answer LLM drops the ANSWER: marker, return raw content."""
        engine = TreeOfThoughtsEngine(config={"n_branches": 1, "beam_width": 1, "max_depth": 1})
        engine.set_brain(_make_brain([
            "A thought.",
            "5\nOk.",
            "Just the answer text without marker.",
        ]))
        result = engine.continue_chat([_user("q")])
        self.assertEqual(result.content, "Just the answer text without marker.")


# ===========================================================================
# Factory smoke tests
# ===========================================================================

class TestNewFactoryPlugins(unittest.TestCase):
    def test_importable(self):
        from ovos_agentic_loop.factory import (  # noqa: F401
            ChainOfThoughtEnginePlugin,
            CRITICEnginePlugin,
            TreeOfThoughtsEnginePlugin,
        )

    def test_subclasses(self):
        from ovos_agentic_loop.factory import (
            ChainOfThoughtEnginePlugin,
            CRITICEnginePlugin,
            TreeOfThoughtsEnginePlugin,
        )
        self.assertTrue(issubclass(ChainOfThoughtEnginePlugin, ChainOfThoughtEngine))
        self.assertTrue(issubclass(CRITICEnginePlugin, CRITICEngine))
        self.assertTrue(issubclass(TreeOfThoughtsEnginePlugin, TreeOfThoughtsEngine))


if __name__ == "__main__":
    unittest.main()
