"""AgenticLoopEngine — base class for agent-loop ChatEngine plugins."""
import abc
from typing import Any, Dict, List, Optional

from ovos_plugin_manager.templates.agents import AgentMessage, ChatEngine


class AgenticLoopEngine(ChatEngine):
    """
    A ``ChatEngine`` subclass for plugins that implement an internal agent loop
    (e.g. ReAct, tool-call/observe cycles, background worker agents).

    From the perspective of a ``PersonaService`` or any caller, an
    ``AgenticLoopEngine`` is identical to a ``ChatEngine`` — it receives a list
    of ``AgentMessage`` objects and returns one.  All loop mechanics, tool
    dispatch, retries, and background tasks are implementation details hidden
    inside the plugin.

    The ``toolboxes`` attribute gives persona configs a standard place to inject
    ``ToolBox`` instances.  Plugins are free to discover and load additional
    toolboxes internally via ``_load_toolboxes_from_config``.

    Entry point group: ``opm.agents.chat``
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialise the engine and optionally load toolboxes from config.

        Args:
            config: Plugin-specific configuration dictionary.  May contain a
                ``"toolboxes"`` key listing toolbox plugin IDs to load.
        """
        super().__init__(config=config)
        self.toolboxes: List[Any] = []  # List[ToolBox] — avoids circular import
        self._load_toolboxes_from_config()

    def load_toolboxes(self, toolboxes: List[Any]) -> None:
        """
        Register a list of ``ToolBox`` instances with this engine.

        Replaces any previously registered toolboxes.

        Args:
            toolboxes: Instantiated ``ToolBox`` objects to make available to
                the agent loop.
        """
        self.toolboxes = list(toolboxes)

    def _load_toolboxes_from_config(self) -> None:
        """
        Discover and instantiate toolboxes declared in ``config["toolboxes"]``.

        Reads a list of toolbox plugin IDs from the plugin config, calls OPM to
        find matching ``ToolBox`` plugins, and populates ``self.toolboxes``.
        Does nothing if the config key is absent or OPM is unavailable.
        """
        toolbox_ids: List[str] = self.config.get("toolboxes", [])
        if not toolbox_ids:
            return
        try:
            from ovos_plugin_manager.agent_tools import find_toolbox_plugin, load_toolbox_plugin
            for tid in toolbox_ids:
                try:
                    plugin = load_toolbox_plugin(tid, config=self.config.get(tid, {}))
                    if plugin is not None:
                        self.toolboxes.append(plugin)
                except Exception:  # noqa: BLE001 — best-effort; log and continue
                    pass
        except ImportError:
            pass

    @abc.abstractmethod
    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None) -> AgentMessage:
        """
        Run the agent loop and return the final response.

        The implementation is responsible for all internal steps: tool
        selection, execution, observation, and iteration.  The caller always
        receives a single ``AgentMessage`` with ``MessageRole.ASSISTANT``.

        Args:
            messages: Full conversation history including the latest user turn.
            session_id: Conversation session identifier.
            lang: BCP-47 language code.
            units: Preferred measurement system (``"metric"`` / ``"imperial"``).

        Returns:
            The assistant's final response after the loop has completed.
        """
        raise NotImplementedError()
