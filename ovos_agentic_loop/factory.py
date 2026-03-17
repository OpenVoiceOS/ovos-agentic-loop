"""OPM entry-point factory classes for ovos-agentic-loop."""
from ovos_agentic_loop.react import ReActLoopEngine

# Re-export so pyproject.toml entry points can reference this module.
__all__ = ["ReActLoopEnginePlugin"]


class ReActLoopEnginePlugin(ReActLoopEngine):
    """
    OPM-registered plugin class for the ReAct loop engine.

    Entry point group: ``opm.agents.chat``
    Entry point name:  ``ovos-react-loop``
    """
