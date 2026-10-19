"""A loop engine loads its brain from config through the real OPM plugin loader.

No mock stands in for ``ovos_plugin_manager.agents``: the brain is the installed
``ovos-solver-failure-plugin`` entry point, resolved and built the way a persona
config resolves it.
"""
import pytest
from ovos_solver_failure_plugin import FailureChatEngine

from ovos_agentic_loop.factory import (
    ChainOfThoughtEnginePlugin,
    CRITICEnginePlugin,
    NativeToolCallEnginePlugin,
    PlanAndExecuteEnginePlugin,
    ReActLoopEnginePlugin,
    ReflexionEnginePlugin,
    SelfAskEnginePlugin,
    TreeOfThoughtsEnginePlugin,
)

ENGINES = [
    ChainOfThoughtEnginePlugin,
    CRITICEnginePlugin,
    NativeToolCallEnginePlugin,
    PlanAndExecuteEnginePlugin,
    ReActLoopEnginePlugin,
    ReflexionEnginePlugin,
    SelfAskEnginePlugin,
    TreeOfThoughtsEnginePlugin,
]

BRAIN_ID = "ovos-solver-failure-plugin"


@pytest.mark.parametrize("engine_cls", ENGINES, ids=lambda c: c.__name__)
def test_brain_is_an_instance_built_with_its_own_config(engine_cls):
    brain_config = {"marker": engine_cls.__name__}
    engine = engine_cls({"brain": BRAIN_ID, BRAIN_ID: brain_config})

    brain = engine.brain

    assert isinstance(brain, FailureChatEngine)
    assert brain.config == brain_config
