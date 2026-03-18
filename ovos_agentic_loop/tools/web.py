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

"""WebSearchToolBox — DuckDuckGo web search as an agent tool."""
from typing import Any, Dict, List, Optional

from ovos_plugin_manager.templates.agent_tools import AgentTool, ToolArguments, ToolBox, ToolOutput
from pydantic import Field


class WebSearchArgs(ToolArguments):
    """Arguments for ``web_search``."""

    query: str = Field(..., description="Search query string.")
    max_results: int = Field(5, description="Maximum number of results to return.")


class WebSearchOutput(ToolOutput):
    """Output of ``web_search``."""

    results: List[Dict[str, str]] = Field(
        ...,
        description="List of results, each with keys: title, url, snippet.",
    )
    query: str = Field(..., description="The query that was executed.")


class WebSearchToolBox(ToolBox):
    """
    A ``ToolBox`` plugin that exposes DuckDuckGo web search as an agent tool.

    Requires the ``duckduckgo-search`` package (``pip install duckduckgo-search``).
    If the package is not installed the tool returns a friendly error rather than
    crashing the agent loop.

    Entry point group: ``opm.agents.toolbox``

    Optional dependency:
    - ``duckduckgo-search>=6.0`` (install via ``pip install ovos-agentic-loop[web]``)
    """

    toolbox_id = "ovos-web-search-tools"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialise the toolbox.

        Args:
            config: Plugin configuration dict (currently unused).
        """
        self.config: Dict[str, Any] = config or {}
        super().__init__(toolbox_id=self.toolbox_id)

    def _web_search(self, args: WebSearchArgs) -> WebSearchOutput:
        """
        Perform a DuckDuckGo text search and return structured results.

        Falls back to a friendly error if ``duckduckgo-search`` is not installed.

        Args:
            args: Validated ``WebSearchArgs``.

        Returns:
            ``WebSearchOutput`` with up to ``max_results`` results.
        """
        try:
            from duckduckgo_search import DDGS  # type: ignore[import-untyped]
        except ImportError:
            return WebSearchOutput(
                results=[{
                    "title": "Package not installed",
                    "url": "",
                    "snippet": (
                        "The 'duckduckgo-search' package is required for web search. "
                        "Install it with: pip install 'ovos-agentic-loop[web]'"
                    ),
                }],
                query=args.query,
            )

        try:
            with DDGS() as ddgs:
                raw = ddgs.text(args.query, max_results=args.max_results)
            results = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("url", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                }
                for r in (raw or [])
            ]
        except Exception as exc:
            results = [{
                "title": "Search error",
                "url": "",
                "snippet": f"Web search failed: {exc}",
            }]

        return WebSearchOutput(results=results, query=args.query)

    def discover_tools(self) -> List[AgentTool]:
        """
        Return the web search ``AgentTool`` instances.

        Returns:
            List containing the ``web_search`` tool.
        """
        return [
            AgentTool(
                name="web_search",
                description=(
                    "Search the web using DuckDuckGo and return a list of results "
                    "with title, URL, and snippet for each."
                ),
                argument_schema=WebSearchArgs,
                output_schema=WebSearchOutput,
                tool_call=self._web_search,
            )
        ]
