"""ClockToolBox — current date/time as an agent tool."""
from datetime import datetime
from typing import Any, Dict, List, Optional

from ovos_plugin_manager.templates.agent_tools import AgentTool, ToolArguments, ToolBox, ToolOutput
from pydantic import Field


class GetCurrentDatetimeArgs(ToolArguments):
    """Arguments for ``get_current_datetime`` (none required)."""


class GetCurrentDatetimeOutput(ToolOutput):
    """Output of ``get_current_datetime``."""

    iso: str = Field(..., description="Full ISO-8601 datetime string.")
    date: str = Field(..., description="Date portion (YYYY-MM-DD).")
    time: str = Field(..., description="Time portion (HH:MM:SS).")
    timezone: str = Field(..., description="System timezone name or offset.")


class ClockToolBox(ToolBox):
    """
    A ``ToolBox`` plugin that exposes the current system date and time.

    No external dependencies required.

    Entry point group: ``opm.agents.toolbox``
    """

    toolbox_id = "ovos-clock-tools"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialise the toolbox.

        Args:
            config: Plugin configuration dict (currently unused).
        """
        self.config: Dict[str, Any] = config or {}
        super().__init__(toolbox_id=self.toolbox_id)

    def _get_current_datetime(self, args: GetCurrentDatetimeArgs) -> GetCurrentDatetimeOutput:
        """
        Return the current system date and time.

        Args:
            args: Validated ``GetCurrentDatetimeArgs`` (no fields).

        Returns:
            ``GetCurrentDatetimeOutput`` with ISO string, date, time, and timezone.
        """
        now = datetime.now().astimezone()
        tz_name = now.tzname() or str(now.utcoffset())
        return GetCurrentDatetimeOutput(
            iso=now.isoformat(),
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H:%M:%S"),
            timezone=tz_name,
        )

    def discover_tools(self) -> List[AgentTool]:
        """
        Return the clock ``AgentTool`` instances.

        Returns:
            List containing the ``get_current_datetime`` tool.
        """
        return [
            AgentTool(
                name="get_current_datetime",
                description="Return the current system date and time in ISO-8601 format.",
                argument_schema=GetCurrentDatetimeArgs,
                output_schema=GetCurrentDatetimeOutput,
                tool_call=self._get_current_datetime,
            )
        ]
