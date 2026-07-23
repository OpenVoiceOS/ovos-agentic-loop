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

"""SkillMDToolBox — exposes each SKILL.md as an AgentTool via a sub-LLM call."""
import re
from typing import Any, Dict, List, Optional

from ovos_plugin_manager.templates.agent_tools import AgentTool, ToolArguments, ToolBox, ToolOutput
from ovos_plugin_manager.templates.agents import AgentMessage, ChatEngine, MessageRole
from pydantic import Field

from ovos_agentic_loop.skills.loader import SkillMDEntry, SkillMDLoader


def _slugify(name: str) -> str:
    """
    Convert a skill name to a valid tool identifier.

    Replaces non-alphanumeric characters with underscores and lower-cases.

    Args:
        name: Raw skill name from SKILL.md frontmatter.

    Returns:
        Slug suitable for use as an ``AgentTool.name``.
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


class SkillCallArgs(ToolArguments):
    """
    Arguments for invoking a SKILL.md-backed tool.

    The I/O contract is intentionally natural-language — the ``task`` is a
    free-text instruction that the sub-LLM (guided by the SKILL.md body) will
    fulfil.
    """
    task: str = Field(..., description="What you want the skill to do.")
    context: Optional[str] = Field(
        None,
        description="Additional context or constraints for the skill.",
    )


class SkillCallOutput(ToolOutput):
    """Output returned by a SKILL.md-backed tool invocation."""
    result: str = Field(..., description="The skill's natural-language response.")
    skill_used: str = Field(..., description="Name of the skill that produced the result.")


class SkillMDToolBox(ToolBox):
    """
    A ``ToolBox`` plugin that converts installed ``SKILL.md`` files into
    ``AgentTool`` instances callable by an agentic loop engine.

    Each discovered SKILL.md yields one tool:

    - ``AgentTool.name`` — slugified from the frontmatter ``name`` field.
    - ``AgentTool.description`` — verbatim from the frontmatter ``description`` field.
    - Tool execution — calls a sub-``ChatEngine`` with the SKILL.md body as
      the system prompt and the ``task`` argument as the user message.

    The sub-ChatEngine (``brain``) is either injected via ``set_brain()`` or
    supplied at construction time.

    Entry point group: ``opm.agents.toolbox``

    Config keys:
    - ``extra_skill_md_paths`` (List[str]): Additional SKILL.md paths to load.
    """

    toolbox_id = "ovos-skill-md-toolbox"

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 bus: Optional[Any] = None,
                 brain: Optional[ChatEngine] = None) -> None:
        """
        Initialise the toolbox.

        Args:
            config: Plugin configuration dict.
            bus: Optional message bus connection, forwarded to the base class.
            brain: ChatEngine used to execute skill invocations.  Must be set
                before any tool is called.
        """
        super().__init__(config=config, bus=bus)
        self._brain: Optional[ChatEngine] = brain
        extra = self.config.get("extra_skill_md_paths", [])
        self._loader = SkillMDLoader(extra_paths=extra)

    def set_brain(self, brain: ChatEngine) -> None:
        """
        Inject the inner ChatEngine used for skill execution.

        Args:
            brain: Instantiated ``ChatEngine``.
        """
        self._brain = brain

    def _invoke_skill(self, entry: SkillMDEntry, args: SkillCallArgs) -> SkillCallOutput:
        """
        Execute a single skill by calling the brain with the SKILL.md body as
        its system prompt.

        Args:
            entry: The parsed ``SkillMDEntry`` whose body becomes the system
                prompt.
            args: Validated ``SkillCallArgs`` from the agent loop.

        Returns:
            ``SkillCallOutput`` with the brain's response.

        Raises:
            RuntimeError: If no brain has been configured.
        """
        if self._brain is None:
            raise RuntimeError(
                "SkillMDToolBox: brain ChatEngine is not configured. "
                "Call set_brain() before invoking tools."
            )
        user_content = args.task
        if args.context:
            user_content = f"{args.task}\n\nContext: {args.context}"

        messages: List[AgentMessage] = [
            AgentMessage(role=MessageRole.SYSTEM, content=entry.body),
            AgentMessage(role=MessageRole.USER, content=user_content),
        ]
        response = self._brain.continue_chat(messages)
        return SkillCallOutput(result=response.content, skill_used=entry.name)

    def discover_tools(self) -> List[AgentTool]:
        """
        Discover all installed SKILL.md files and build an ``AgentTool`` for each.

        Returns:
            List of ``AgentTool`` instances, one per valid SKILL.md file.
        """
        tools: List[AgentTool] = []
        for entry in self._loader.load():
            # Capture entry in closure for each iteration.
            _entry = entry
            tools.append(AgentTool(
                name=_slugify(_entry.name),
                description=_entry.description,
                argument_schema=SkillCallArgs,
                output_schema=SkillCallOutput,
                tool_call=lambda args, e=_entry: self._invoke_skill(e, args),
            ))
        return tools
