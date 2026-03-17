"""ShellToolBox — execute shell commands as an agent tool."""
import subprocess
from typing import Any, Dict, List, Optional

from ovos_plugin_manager.templates.agent_tools import AgentTool, ToolArguments, ToolBox, ToolOutput
from pydantic import Field


class RunCommandArgs(ToolArguments):
    """Arguments for ``run_command``."""

    command: str = Field(..., description="Shell command to execute.")
    cwd: str = Field(".", description="Working directory for the command.")
    timeout: int = Field(30, description="Timeout in seconds (capped by max_timeout config).")


class RunCommandOutput(ToolOutput):
    """Output of ``run_command``."""

    stdout: str = Field(..., description="Standard output from the command.")
    stderr: str = Field(..., description="Standard error from the command.")
    returncode: int = Field(..., description="Exit code of the process.")
    success: bool = Field(..., description="True if returncode == 0.")


class ShellToolBox(ToolBox):
    """
    A ``ToolBox`` plugin that exposes shell command execution as an agent tool.

    Entry point group: ``opm.agents.toolbox``

    Config keys:
    - ``allow_shell`` (bool, default ``True``): Set to ``False`` to disable execution.
    - ``max_timeout`` (int, default ``120``): Maximum timeout in seconds.
    """

    toolbox_id = "ovos-shell-tools"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialise the toolbox.

        Args:
            config: Plugin configuration dict.  Recognised keys:
                ``allow_shell`` (bool, default ``True``),
                ``max_timeout`` (int, default ``120``).
        """
        self.config: Dict[str, Any] = config or {}
        super().__init__(toolbox_id=self.toolbox_id)

    def _run_command(self, args: RunCommandArgs) -> RunCommandOutput:
        """
        Execute a shell command and return its stdout, stderr, and exit code.

        The requested timeout is capped at ``config["max_timeout"]`` (default 120 s).

        Args:
            args: Validated ``RunCommandArgs``.

        Returns:
            ``RunCommandOutput`` with execution results.
        """
        if not self.config.get("allow_shell", True):
            return RunCommandOutput(
                stdout="",
                stderr="Shell execution is disabled (allow_shell=False).",
                returncode=-1,
                success=False,
            )

        max_timeout = int(self.config.get("max_timeout", 120))
        effective_timeout = min(args.timeout, max_timeout)

        try:
            result = subprocess.run(
                args.command,
                shell=True,
                cwd=args.cwd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
            return RunCommandOutput(
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
                success=result.returncode == 0,
            )
        except subprocess.TimeoutExpired:
            return RunCommandOutput(
                stdout="",
                stderr=f"Command timed out after {effective_timeout} seconds.",
                returncode=-1,
                success=False,
            )
        except Exception as exc:
            return RunCommandOutput(
                stdout="",
                stderr=f"Error executing command: {exc}",
                returncode=-1,
                success=False,
            )

    def discover_tools(self) -> List[AgentTool]:
        """
        Return the shell ``AgentTool`` instances.

        Returns:
            List containing the ``run_command`` tool.
        """
        return [
            AgentTool(
                name="run_command",
                description="Execute a shell command and return its stdout, stderr, and exit code.",
                argument_schema=RunCommandArgs,
                output_schema=RunCommandOutput,
                tool_call=self._run_command,
            )
        ]
