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
"""Tests for PlanAndExecuteEngine, ReflexionEngine, SelfAskEngine."""
import unittest
from unittest.mock import MagicMock, call, patch
from typing import List

from ovos_plugin_manager.templates.agents import AgentMessage, MessageRole

from ovos_agentic_loop.plan_execute import (
    PlanAndExecuteEngine,
    _parse_plan,
    _extract_step_result,
)
from ovos_agentic_loop.reflexion import ReflexionEngine
from ovos_agentic_loop.self_ask import (
    SelfAskEngine,
    _extract_final_answer,
    _extract_follow_up,
    _extract_tool_call,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_brain(responses: List[str]) -> MagicMock:
    """Create a mock ChatEngine that returns ``responses`` in order."""
    brain = MagicMock()
    brain.continue_chat.side_effect = [
        AgentMessage(role=MessageRole.ASSISTANT, content=r)
        for r in responses
    ]
    return brain


def _user(text: str) -> AgentMessage:
    return AgentMessage(role=MessageRole.USER, content=text)


# ---------------------------------------------------------------------------
# PlanAndExecuteEngine helpers
# ---------------------------------------------------------------------------

class TestParsePlan(unittest.TestCase):
    def test_numbered_dot(self):
        text = "1. Get weather\n2. Check forecast\n3. Summarise"
        self.assertEqual(_parse_plan(text), ["Get weather", "Check forecast", "Summarise"])

    def test_numbered_paren(self):
        text = "1) First step\n2) Second step"
        self.assertEqual(_parse_plan(text), ["First step", "Second step"])

    def test_ignores_prose(self):
        text = "Here is your plan:\n1. Do A\nSome filler text\n2. Do B"
        self.assertEqual(_parse_plan(text), ["Do A", "Do B"])

    def test_empty(self):
        self.assertEqual(_parse_plan("No numbered steps here."), [])


class TestExtractStepResult(unittest.TestCase):
    def test_found(self):
        text = "I called the tool.\nSTEP_RESULT: Paris is 15°C"
        self.assertEqual(_extract_step_result(text), "Paris is 15°C")

    def test_not_found(self):
        self.assertIsNone(_extract_step_result("No result token here."))


# ---------------------------------------------------------------------------
# PlanAndExecuteEngine integration
# ---------------------------------------------------------------------------

class TestPlanAndExecuteEngine(unittest.TestCase):
    def _engine(self, responses: List[str]) -> PlanAndExecuteEngine:
        engine = PlanAndExecuteEngine(config={})
        engine.set_brain(_make_brain(responses))
        return engine

    def test_no_brain_returns_error(self):
        engine = PlanAndExecuteEngine(config={})
        result = engine.continue_chat([_user("hello")])
        self.assertIn("Error", result.content)

    def test_full_plan_execute_synthesize(self):
        """Plan → 2 steps → synthesize."""
        engine = self._engine([
            "1. Get weather in Paris\n2. Summarise the answer",   # planner
            "STEP_RESULT: Paris is sunny, 18°C",                   # step 1 executor
            "STEP_RESULT: Summary done",                            # step 2 executor
            "Paris is sunny at 18°C today.",                        # synthesizer
        ])
        result = engine.continue_chat([_user("What is the weather in Paris?")])
        self.assertEqual(result.content, "Paris is sunny at 18°C today.")
        self.assertEqual(result.role, MessageRole.ASSISTANT)

    def test_fallback_when_no_parseable_plan(self):
        """When the planner returns unparseable text it is used as-is."""
        engine = self._engine(["I don't know how to make a plan for that."])
        result = engine.continue_chat([_user("What is 2+2?")])
        self.assertEqual(result.content, "I don't know how to make a plan for that.")

    def test_max_steps_cap(self):
        """Engine respects max_steps and does not execute more steps than allowed."""
        plan_text = "\n".join(f"{i}. Step {i}" for i in range(1, 9))  # 8 steps
        step_responses = [f"STEP_RESULT: done {i}" for i in range(1, 6)]  # 5 results
        engine = PlanAndExecuteEngine(config={"max_steps": 5})
        engine.set_brain(_make_brain([plan_text, *step_responses, "Final answer."]))
        result = engine.continue_chat([_user("do 8 things")])
        self.assertEqual(result.content, "Final answer.")

    def test_step_with_tool_call(self):
        """A step executor that emits Action/Action Input triggers a tool call."""
        engine = PlanAndExecuteEngine(config={})
        engine.set_brain(_make_brain([
            "1. Look up the capital of France",                        # planner
            'Action: search\nAction Input: {"query": "capital France"}\n',  # executor calls tool
            "STEP_RESULT: The capital of France is Paris.",            # executor wraps up
            "The capital of France is Paris.",                          # synthesizer
        ]))
        # Register a mock toolbox.
        tb = MagicMock()
        tb.tool_json_list = [{"name": "search", "description": "search the web"}]
        tb.get_tool.return_value = MagicMock()
        tb.call_tool.return_value = "Paris"
        engine.toolboxes = [tb]

        result = engine.continue_chat([_user("What is the capital of France?")])
        self.assertEqual(result.content, "The capital of France is Paris.")
        tb.call_tool.assert_called_once_with("search", {"query": "capital France"})


# ---------------------------------------------------------------------------
# ReflexionEngine
# ---------------------------------------------------------------------------

class TestReflexionEngine(unittest.TestCase):
    def _engine(self, responses: List[str]) -> ReflexionEngine:
        engine = ReflexionEngine(config={})
        engine.set_brain(_make_brain(responses))
        return engine

    def test_no_brain_returns_error(self):
        engine = ReflexionEngine(config={})
        result = engine.continue_chat([_user("hello")])
        self.assertIn("Error", result.content)

    def test_satisfactory_on_first_attempt(self):
        """If evaluator says SATISFACTORY, return immediately without reflecting."""
        engine = self._engine([
            "Paris is 18°C and sunny.",           # episode 1 ReAct inner answer
            "SATISFACTORY\nAnswer is complete.",  # evaluator
        ])
        result = engine.continue_chat([_user("What is the weather in Paris?")])
        self.assertEqual(result.content, "Paris is 18°C and sunny.")
        # Only 2 brain calls: react answer + evaluator.
        self.assertEqual(engine.brain.continue_chat.call_count, 2)

    def test_reflection_on_failure_then_success(self):
        """UNSATISFACTORY triggers reflection, second episode succeeds."""
        engine = self._engine([
            "I don't know.",                         # episode 1 answer
            "UNSATISFACTORY\nDid not use tools.",    # evaluator
            "Reflection: I should call the weather tool next time.",  # reflector
            "Paris is 18°C.",                        # episode 2 answer
            "SATISFACTORY\nAnswer is complete.",     # evaluator 2
        ])
        result = engine.continue_chat([_user("What is the weather in Paris?")])
        self.assertEqual(result.content, "Paris is 18°C.")

    def test_max_reflections_returns_last_answer(self):
        """After max_reflections episodes, return last answer regardless of eval."""
        engine = ReflexionEngine(config={"max_reflections": 2})
        engine.set_brain(_make_brain([
            "Bad answer 1.",
            "UNSATISFACTORY\nNot good.",
            "Reflection: try harder.",
            "Bad answer 2.",
            "UNSATISFACTORY\nStill not good.",
        ]))
        result = engine.continue_chat([_user("question")])
        self.assertEqual(result.content, "Bad answer 2.")

    def test_inner_react_shares_toolboxes(self):
        """Toolboxes registered on ReflexionEngine are passed to inner ReAct."""
        engine = ReflexionEngine(config={})
        engine.set_brain(_make_brain(["answer", "SATISFACTORY\nok"]))
        tb = MagicMock()
        tb.tool_json_list = []
        engine.load_toolboxes([tb])
        react = engine._get_react_engine()
        self.assertIn(tb, react.toolboxes)


# ---------------------------------------------------------------------------
# SelfAskEngine helpers
# ---------------------------------------------------------------------------

class TestSelfAskHelpers(unittest.TestCase):
    def test_extract_final_answer(self):
        text = "I now have all facts.\nSo the final answer is: Tim Cook."
        self.assertEqual(_extract_final_answer(text), "Tim Cook.")

    def test_extract_final_answer_missing(self):
        self.assertIsNone(_extract_final_answer("No final here."))

    def test_extract_follow_up(self):
        text = "Are follow up questions needed? Yes.\nFollow up: Who won in 2022?"
        self.assertEqual(_extract_follow_up(text), "Who won in 2022?")

    def test_extract_follow_up_missing(self):
        self.assertIsNone(_extract_follow_up("No follow up needed."))

    def test_extract_tool_call(self):
        text = "I need to search.\nTool: web_search\nTool Input: FIFA 2022 winner"
        self.assertEqual(_extract_tool_call(text), ("web_search", "FIFA 2022 winner"))

    def test_extract_tool_call_missing(self):
        self.assertIsNone(_extract_tool_call("No tool call here."))


# ---------------------------------------------------------------------------
# SelfAskEngine integration
# ---------------------------------------------------------------------------

class TestSelfAskEngine(unittest.TestCase):
    def _engine(self, responses: List[str]) -> SelfAskEngine:
        engine = SelfAskEngine(config={})
        engine.set_brain(_make_brain(responses))
        return engine

    def test_no_brain_returns_error(self):
        engine = SelfAskEngine(config={})
        result = engine.continue_chat([_user("hello")])
        self.assertIn("Error", result.content)

    def test_direct_answer_no_followup(self):
        """If the LLM says 'No follow up needed' and gives final answer directly."""
        engine = self._engine([
            "Are follow up questions needed here? No.\n"
            "So the final answer is: 42."
        ])
        result = engine.continue_chat([_user("What is 6*7?")])
        self.assertEqual(result.content, "42.")

    def test_single_followup_with_tool(self):
        """One Follow up: triggers a tool call, then final answer."""
        engine = self._engine([
            "Are follow up questions needed here? Yes.\nFollow up: Who won FIFA 2022?",
            "So the final answer is: Argentina.",
        ])
        mock_tool = MagicMock()
        mock_tool.name = "web_search"
        tb = MagicMock()
        tb.discover_tools.return_value = [mock_tool]
        tb.call_tool.return_value = "Argentina won FIFA 2022."
        engine.toolboxes = [tb]

        result = engine.continue_chat([_user("Who is president of the FIFA 2022 champion?")])
        self.assertEqual(result.content, "Argentina.")
        tb.call_tool.assert_called_once()

    def test_explicit_tool_block(self):
        """Tool: / Tool Input: block triggers named tool call."""
        engine = self._engine([
            "I need a lookup.\nTool: lookup\nTool Input: capital of France",
            "So the final answer is: Paris.",
        ])
        tb = MagicMock()
        tb.get_tool.return_value = MagicMock()
        tb.call_tool.return_value = "Paris"
        engine.toolboxes = [tb]

        result = engine.continue_chat([_user("What is the capital of France?")])
        self.assertEqual(result.content, "Paris.")
        tb.call_tool.assert_called_once_with("lookup", {"query": "capital of France"})

    def test_max_follow_ups_forces_answer(self):
        """After max_follow_ups iterations, a final answer is forced."""
        # All responses request another follow up; last response gives final answer.
        follow_up_resp = "Are follow up questions needed here? Yes.\nFollow up: sub-question?"
        engine = SelfAskEngine(config={"max_follow_ups": 3})
        engine.set_brain(_make_brain([
            follow_up_resp,
            follow_up_resp,
            follow_up_resp,
            "So the final answer is: forced.",
        ]))
        mock_tool = MagicMock()
        mock_tool.name = "search"
        tb = MagicMock()
        tb.discover_tools.return_value = [mock_tool]
        tb.call_tool.return_value = "some result"
        engine.toolboxes = [tb]

        result = engine.continue_chat([_user("complex question")])
        self.assertEqual(result.content, "forced.")

    def test_no_tools_no_follow_up_loop(self):
        """Without tools, follow-up cannot be dispatched; the raw LLM response is returned."""
        first_response = "Are follow up questions needed? Yes.\nFollow up: What is 2+2?"
        engine = self._engine([first_response])
        # No toolboxes registered — follow-up can't be dispatched, so the
        # first response is returned as-is.
        result = engine.continue_chat([_user("What is 2+2?")])
        self.assertEqual(result.content, first_response)

    def test_unrecognized_pattern_returns_as_final(self):
        """If LLM produces unrecognised output without follow-up/final token, return it."""
        engine = self._engine(["Some unstructured response from the LLM."])
        result = engine.continue_chat([_user("hm")])
        self.assertEqual(result.content, "Some unstructured response from the LLM.")


# ---------------------------------------------------------------------------
# Factory / entry-points smoke test
# ---------------------------------------------------------------------------

class TestFactoryImports(unittest.TestCase):
    def test_all_plugins_importable(self):
        from ovos_agentic_loop.factory import (  # noqa: F401
            PlanAndExecuteEnginePlugin,
            ReActLoopEnginePlugin,
            ReflexionEnginePlugin,
            SelfAskEnginePlugin,
        )

    def test_plugin_subclasses(self):
        from ovos_agentic_loop.factory import (
            PlanAndExecuteEnginePlugin,
            ReflexionEnginePlugin,
            SelfAskEnginePlugin,
        )
        self.assertTrue(issubclass(PlanAndExecuteEnginePlugin, PlanAndExecuteEngine))
        self.assertTrue(issubclass(ReflexionEnginePlugin, ReflexionEngine))
        self.assertTrue(issubclass(SelfAskEnginePlugin, SelfAskEngine))


if __name__ == "__main__":
    unittest.main()
