"""AgentsMDContextManager — builds agent system prompts from AGENTS.md files."""
import importlib.metadata
import os
import re
from typing import Dict, List, Optional

from ovos_plugin_manager.templates.agents import AgentContextManager, AgentMessage, MessageRole
from ovos_utils.log import LOG


def _parse_sections(text: str) -> Dict[str, str]:
    """
    Parse a markdown document into a dict mapping heading titles to section bodies.

    Headings at any level (``#``, ``##``, etc.) are recognised.  The section
    body is everything from the heading line up to (but not including) the next
    heading of the same or higher level.

    Args:
        text: Full markdown document text.

    Returns:
        ``{heading_title: section_body_text}`` — headings are stripped of
        leading ``#`` characters and whitespace.
    """
    sections: Dict[str, str] = {}
    pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[heading] = text[start:end].strip()

    return sections


def _discover_agents_md_paths() -> List[str]:
    """
    Discover AGENTS.md files from installed package data.

    Walks every installed distribution's recorded files looking for files
    named ``AGENTS.md``.

    Returns:
        List of absolute paths that exist on disk.
    """
    paths: List[str] = []
    try:
        dists = importlib.metadata.distributions()
    except Exception as exc:  # noqa: BLE001
        LOG.debug(f"AgentsMDContextManager: could not enumerate distributions: {exc}")
        return paths

    for dist in dists:
        try:
            files = dist.files or []
            for f in files:
                if f.name == "AGENTS.md":
                    abs_path = str(f.locate().resolve())
                    if os.path.isfile(abs_path):
                        paths.append(abs_path)
        except Exception as exc:  # noqa: BLE001
            LOG.debug(f"AgentsMDContextManager: error scanning dist files: {exc}")
            continue
    return paths


class AgentsMDContextManager(AgentContextManager):
    """
    An ``AgentContextManager`` that assembles the agent system prompt by
    loading selected sections from one or more ``AGENTS.md`` files.

    The same AGENTS.md that governs Claude Code at dev-time can govern a
    runtime LLM agent — both consumers read the same document.

    Per-session conversation history is stored in an in-memory dict keyed by
    ``session_id``.

    Discovery:

    1. Config ``agents_md_sources`` — list of paths or ``["auto"]`` to trigger
       automatic package-data discovery.
    2. ``extra_paths`` constructor argument — additional ad-hoc paths.

    Config keys:

    - ``agents_md_sources`` (List[str]): Paths to load, or ``["auto"]``.
    - ``include_sections`` (List[str]): Heading titles to include (substring
      match).  If empty, all sections are included.
    - ``system_prompt_prefix`` (str): Text prepended before the assembled
      AGENTS.md content.

    Entry point group: ``opm.agents.memory``
    """

    plugin_id = "ovos-agents-md-context-plugin"

    def __init__(self, config: Optional[Dict] = None,
                 extra_paths: Optional[List[str]] = None) -> None:
        """
        Initialise the context manager.

        Args:
            config: Plugin configuration dictionary.
            extra_paths: Additional AGENTS.md file paths to load.
        """
        super().__init__(config=config)
        self._extra_paths: List[str] = list(extra_paths or [])
        # Per-session history: {session_id: [AgentMessage, ...]}
        self._sessions: Dict[str, List[AgentMessage]] = {}
        self._system_prompt_cache: Optional[str] = None

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    @property
    def include_sections(self) -> List[str]:
        """Heading substrings that filter which sections are included."""
        return list((self.config or {}).get("include_sections", []))

    @property
    def system_prompt_prefix(self) -> str:
        """Text prepended before assembled AGENTS.md content."""
        return str((self.config or {}).get("system_prompt_prefix", ""))

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve_paths(self) -> List[str]:
        """
        Return deduplicated list of AGENTS.md paths to load.

        Returns:
            Absolute path list.
        """
        sources: List[str] = list((self.config or {}).get("agents_md_sources", ["auto"]))
        paths: List[str] = []

        if "auto" in sources:
            paths.extend(_discover_agents_md_paths())
        for src in sources:
            if src != "auto" and os.path.isfile(src):
                paths.append(src)
        paths.extend(self._extra_paths)

        # Deduplicate while preserving order.
        seen: set = set()
        result: List[str] = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                result.append(p)
        return result

    # ------------------------------------------------------------------
    # System prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """
        Load all AGENTS.md files, filter sections, and concatenate.

        Returns:
            The assembled system prompt string.
        """
        parts: List[str] = []
        if self.system_prompt_prefix:
            parts.append(self.system_prompt_prefix)

        for path in self._resolve_paths():
            try:
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
            except OSError as exc:
                LOG.warning(f"AgentsMDContextManager: could not read '{path}': {exc}")
                continue

            sections = _parse_sections(text)
            for heading, body in sections.items():
                if self.include_sections:
                    if not any(inc.lower() in heading.lower()
                               for inc in self.include_sections):
                        continue
                parts.append(f"## {heading}\n\n{body}")

        return "\n\n".join(parts)

    @property
    def system_prompt(self) -> str:
        """
        Assembled system prompt derived from AGENTS.md sections.

        Lazily computed and cached after the first access.
        """
        if self._system_prompt_cache is None:
            self._system_prompt_cache = self._build_system_prompt()
        return self._system_prompt_cache

    def invalidate_cache(self) -> None:
        """Force the system prompt to be rebuilt on next access."""
        self._system_prompt_cache = None

    # ------------------------------------------------------------------
    # AgentContextManager protocol (full ABC compliance)
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> List[AgentMessage]:
        """
        Retrieve the message history for a given session.

        Args:
            session_id: Conversation session identifier.

        Returns:
            Snapshot of the session's message list in chronological order.
        """
        return list(self._sessions.get(session_id, []))

    def update_history(self, new_messages: List[AgentMessage], session_id: str) -> None:
        """
        Append new messages to a session's history.

        Args:
            new_messages: Messages to append.
            session_id: Conversation session identifier.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].extend(new_messages)

    def build_conversation_context(
        self,
        utterance: str,
        session_id: str,
    ) -> List[AgentMessage]:
        """
        Assemble the full message list for an LLM call.

        Prepends the system prompt (assembled from AGENTS.md), appends the
        session's conversation history, and adds the current user utterance
        as the final message.

        Args:
            utterance: The user's current input.
            session_id: Conversation session identifier used to retrieve history.

        Returns:
            Ordered list of ``AgentMessage`` objects ready for a ChatEngine.
        """
        messages: List[AgentMessage] = []

        prompt = self.system_prompt
        if prompt:
            messages.append(AgentMessage(role=MessageRole.SYSTEM, content=prompt))

        messages.extend(self.get_history(session_id))
        messages.append(AgentMessage(role=MessageRole.USER, content=utterance))
        return messages
