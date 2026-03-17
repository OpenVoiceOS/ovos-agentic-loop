"""OPM entry-point factory classes for ovos-agentic-loop."""
from ovos_agentic_loop.plan_execute import PlanAndExecuteEngine
from ovos_agentic_loop.react import ReActLoopEngine
from ovos_agentic_loop.reflexion import ReflexionEngine
from ovos_agentic_loop.self_ask import SelfAskEngine

__all__ = [
    "ReActLoopEnginePlugin",
    "PlanAndExecuteEnginePlugin",
    "ReflexionEnginePlugin",
    "SelfAskEnginePlugin",
]


class ReActLoopEnginePlugin(ReActLoopEngine):
    """
    OPM-registered plugin class for the ReAct loop engine.

    Entry point group: ``opm.agents.chat``
    Entry point name:  ``ovos-react-loop``
    """


class PlanAndExecuteEnginePlugin(PlanAndExecuteEngine):
    """
    OPM-registered plugin class for the Plan-and-Execute engine.

    Entry point group: ``opm.agents.chat``
    Entry point name:  ``ovos-plan-execute-loop``
    """


class ReflexionEnginePlugin(ReflexionEngine):
    """
    OPM-registered plugin class for the Reflexion engine.

    Entry point group: ``opm.agents.chat``
    Entry point name:  ``ovos-reflexion-loop``
    """


class SelfAskEnginePlugin(SelfAskEngine):
    """
    OPM-registered plugin class for the Self-Ask engine.

    Entry point group: ``opm.agents.chat``
    Entry point name:  ``ovos-self-ask-loop``
    """
