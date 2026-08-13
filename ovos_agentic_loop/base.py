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

"""AgenticLoopEngine — base class for agent-loop ChatEngine plugins."""
import abc
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ovos_plugin_manager.templates.agents import AgentMessage, ChatEngine, ToolsArg
from ovos_utils.log import LOG

if TYPE_CHECKING:
    from ovos_plugin_manager.templates.agent_tools import ToolBox


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
        self.toolboxes: "List[ToolBox]" = []
        self._load_toolboxes_from_config()

    def load_toolboxes(self, toolboxes: List[Any]) -> None:
        """
        Register a list of ``ToolBox`` instances with this engine.

        Replaces any previously registered toolboxes.  If the engine already
        has a ``brain`` wired in, it is propagated to toolboxes that expose a
        ``set_brain`` method (e.g. ``SkillMDToolBox``).

        Args:
            toolboxes: Instantiated ``ToolBox`` objects to make available to
                the agent loop.
        """
        self.toolboxes = list(toolboxes)
        brain = getattr(self, "_brain", None)
        if brain is not None:
            self._inject_brain_into_toolboxes(brain)

    def _inject_brain_into_toolboxes(self, brain: Any) -> None:
        """
        Propagate a brain ``ChatEngine`` to toolboxes that require one.

        Calls ``toolbox.set_brain(brain)`` on every registered toolbox that
        exposes a ``set_brain`` method (duck-type check).  This ensures that
        toolboxes like ``SkillMDToolBox`` — which use an inner LLM for
        tool-call routing — receive the brain automatically when it is set on
        the engine, regardless of load order.

        Args:
            brain: The ``ChatEngine`` instance to propagate.
        """
        for tb in self.toolboxes:
            if callable(getattr(tb, "set_brain", None)):
                try:
                    tb.set_brain(brain)
                except Exception as exc:  # noqa: BLE001
                    LOG.warning(
                        f"AgenticLoopEngine: failed to inject brain into "
                        f"{type(tb).__name__}: {exc}"
                    )

    def _load_toolboxes_from_config(self) -> None:
        """
        Discover and instantiate toolboxes declared in ``config["toolboxes"]``.

        Reads a list of toolbox plugin IDs from the plugin config, calls OPM to
        find matching ``ToolBox`` plugins, and populates ``self.toolboxes``.
        Brain injection is deferred — toolboxes are wired after the engine's
        ``_brain`` attribute is set (see ``_inject_brain_into_toolboxes``).
        Does nothing if the config key is absent or OPM is unavailable.
        """
        toolbox_ids: List[str] = self.config.get("toolboxes", [])
        if not toolbox_ids:
            return
        try:
            from ovos_plugin_manager.persona import find_toolbox_plugins
        except ImportError:
            LOG.debug("AgenticLoopEngine: ovos_plugin_manager.persona not available; "
                      "skipping toolbox auto-load")
            return

        bus = getattr(self, "bus", None)
        available = find_toolbox_plugins()
        for tid in toolbox_ids:
            try:
                cls = available[tid]
                self.toolboxes.append(cls(config=self.config.get(tid, {}), bus=bus))
            except Exception as exc:  # noqa: BLE001
                LOG.warning(f"AgenticLoopEngine: failed to load toolbox '{tid}': {exc}")

    @abc.abstractmethod
    def continue_chat(self, messages: List[AgentMessage],
                      session_id: str = "default",
                      lang: Optional[str] = None,
                      units: Optional[str] = None,
                      tools: "ToolsArg" = None) -> AgentMessage:
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
        # `tools` is accepted (and ignored) purely for contract conformance with
        # ovos_plugin_manager.templates.agents.ChatEngine.continue_chat, whose
        # signature declares it unconditionally. This engine is not tool-capable
        # (supports_tools stays False). Accepting the kwarg matters because the
        # agentic-loop ReAct fallback (see native_toolcall.py) calls
        # `self.brain.continue_chat(..., tools=...)` on whatever brain engine is
        # configured, even non-tool-capable ones — omitting `tools` here would
        # raise TypeError on that call path.
        raise NotImplementedError()
