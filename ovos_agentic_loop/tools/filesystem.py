"""FileSystemToolBox — read, write, list, and search files on the local filesystem."""
import glob as _glob
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ovos_plugin_manager.templates.agent_tools import AgentTool, ToolArguments, ToolBox, ToolOutput
from pydantic import Field


class ReadFileArgs(ToolArguments):
    """Arguments for ``read_file``."""

    path: str = Field(..., description="Absolute or relative path of the file to read.")


class ReadFileOutput(ToolOutput):
    """Output of ``read_file``."""

    content: str = Field(..., description="UTF-8 text content of the file, or an error message.")
    path: str = Field(..., description="Resolved path that was read.")


class WriteFileArgs(ToolArguments):
    """Arguments for ``write_file``."""

    path: str = Field(..., description="Absolute or relative path to write.")
    content: str = Field(..., description="UTF-8 text to write into the file.")


class WriteFileOutput(ToolOutput):
    """Output of ``write_file``."""

    success: bool = Field(..., description="True if the write succeeded.")
    path: str = Field(..., description="Resolved path that was written.")
    message: str = Field(..., description="Human-readable status message.")


class ListDirectoryArgs(ToolArguments):
    """Arguments for ``list_directory``."""

    path: str = Field(..., description="Directory path to list.")
    pattern: str = Field("*", description="Shell glob pattern to filter entries.")


class ListDirectoryOutput(ToolOutput):
    """Output of ``list_directory``."""

    entries: List[str] = Field(..., description="Matching filesystem entries.")
    path: str = Field(..., description="Directory that was listed.")


class SearchInFilesArgs(ToolArguments):
    """Arguments for ``search_in_files``."""

    pattern: str = Field(..., description="Regular expression to search for.")
    path: str = Field(".", description="Root directory to search within.")
    glob: str = Field("**/*", description="Glob pattern to select files.")


class SearchInFilesOutput(ToolOutput):
    """Output of ``search_in_files``."""

    matches: List[Dict[str, str]] = Field(
        ...,
        description="List of matches, each with keys: file, line_number, line.",
    )
    total: int = Field(..., description="Total number of matching lines found.")


class FindFilesArgs(ToolArguments):
    """Arguments for ``find_files``."""

    glob: str = Field(..., description="Glob pattern (e.g. '**/*.py').")
    path: str = Field(".", description="Root directory to search from.")


class FindFilesOutput(ToolOutput):
    """Output of ``find_files``."""

    files: List[str] = Field(..., description="Matched file paths as strings.")
    total: int = Field(..., description="Number of files found.")


class FileSystemToolBox(ToolBox):
    """
    A ``ToolBox`` plugin exposing local filesystem operations as agent tools.

    Provides read, write, list, search, and find capabilities.  The ``write_file``
    tool is guarded by the ``allow_write`` config flag.

    Entry point group: ``opm.agents.toolbox``

    Config keys:
    - ``allow_write`` (bool, default ``True``): Set to ``False`` for read-only agents.
    """

    toolbox_id = "ovos-filesystem-tools"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialise the toolbox.

        Args:
            config: Plugin configuration dict.  Recognised key:
                ``allow_write`` (bool, default ``True``).
        """
        self.config: Dict[str, Any] = config or {}
        super().__init__(toolbox_id=self.toolbox_id)

    # --- tool implementations ---

    def _read_file(self, args: ReadFileArgs) -> ReadFileOutput:
        """
        Read a file and return its UTF-8 contents.

        Args:
            args: Validated ``ReadFileArgs``.

        Returns:
            ``ReadFileOutput`` with file content or an error message.
        """
        p = Path(args.path)
        try:
            content = p.read_text(encoding="utf-8")
        except Exception as exc:
            content = f"Error reading file: {exc}"
        return ReadFileOutput(content=content, path=str(p.resolve()))

    def _write_file(self, args: WriteFileArgs) -> WriteFileOutput:
        """
        Write text content to a file, creating parent directories as needed.

        Args:
            args: Validated ``WriteFileArgs``.

        Returns:
            ``WriteFileOutput`` indicating success or failure.
        """
        if not self.config.get("allow_write", True):
            return WriteFileOutput(
                success=False,
                path=args.path,
                message="Write access is disabled (allow_write=False).",
            )
        p = Path(args.path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args.content, encoding="utf-8")
            return WriteFileOutput(success=True, path=str(p.resolve()), message="File written successfully.")
        except Exception as exc:
            return WriteFileOutput(success=False, path=args.path, message=f"Error writing file: {exc}")

    def _list_directory(self, args: ListDirectoryArgs) -> ListDirectoryOutput:
        """
        List directory entries matching a glob pattern.

        Args:
            args: Validated ``ListDirectoryArgs``.

        Returns:
            ``ListDirectoryOutput`` with matching entry names.
        """
        p = Path(args.path)
        try:
            entries = [str(e) for e in sorted(p.glob(args.pattern))]
        except Exception:
            entries = []
        return ListDirectoryOutput(entries=entries, path=str(p.resolve()))

    def _search_in_files(self, args: SearchInFilesArgs) -> SearchInFilesOutput:
        """
        Search for a regex pattern across files matching a glob.

        Args:
            args: Validated ``SearchInFilesArgs``.

        Returns:
            ``SearchInFilesOutput`` with a list of ``{file, line_number, line}`` dicts.
        """
        root = Path(args.path)
        matches: List[Dict[str, str]] = []
        try:
            compiled = re.compile(args.pattern)
        except re.error:
            return SearchInFilesOutput(matches=[], total=0)

        for file_path in root.glob(args.glob):
            if not file_path.is_file():
                continue
            try:
                for lineno, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if compiled.search(line):
                        matches.append({
                            "file": str(file_path),
                            "line_number": str(lineno),
                            "line": line,
                        })
            except Exception:
                continue

        return SearchInFilesOutput(matches=matches, total=len(matches))

    def _find_files(self, args: FindFilesArgs) -> FindFilesOutput:
        """
        Find files under a root directory matching a glob pattern.

        Args:
            args: Validated ``FindFilesArgs``.

        Returns:
            ``FindFilesOutput`` with the list of matching file paths.
        """
        root = Path(args.path)
        files = [str(p) for p in sorted(root.glob(args.glob)) if p.is_file()]
        return FindFilesOutput(files=files, total=len(files))

    def discover_tools(self) -> List[AgentTool]:
        """
        Return the set of filesystem ``AgentTool`` instances.

        Returns:
            List of five ``AgentTool`` objects (read, write, list, search, find).
        """
        tools = [
            AgentTool(
                name="read_file",
                description="Read the UTF-8 text content of a file.",
                argument_schema=ReadFileArgs,
                output_schema=ReadFileOutput,
                tool_call=self._read_file,
            ),
            AgentTool(
                name="list_directory",
                description="List entries in a directory, optionally filtered by a glob pattern.",
                argument_schema=ListDirectoryArgs,
                output_schema=ListDirectoryOutput,
                tool_call=self._list_directory,
            ),
            AgentTool(
                name="search_in_files",
                description="Search for a regex pattern across files under a directory.",
                argument_schema=SearchInFilesArgs,
                output_schema=SearchInFilesOutput,
                tool_call=self._search_in_files,
            ),
            AgentTool(
                name="find_files",
                description="Find files matching a glob pattern under a directory.",
                argument_schema=FindFilesArgs,
                output_schema=FindFilesOutput,
                tool_call=self._find_files,
            ),
        ]
        if self.config.get("allow_write", True):
            tools.insert(1, AgentTool(
                name="write_file",
                description="Write UTF-8 text content to a file, creating parent directories as needed.",
                argument_schema=WriteFileArgs,
                output_schema=WriteFileOutput,
                tool_call=self._write_file,
            ))
        else:
            # Still register the tool but it will return a disabled message.
            tools.insert(1, AgentTool(
                name="write_file",
                description="Write UTF-8 text content to a file (currently disabled — allow_write=False).",
                argument_schema=WriteFileArgs,
                output_schema=WriteFileOutput,
                tool_call=self._write_file,
            ))
        return tools
