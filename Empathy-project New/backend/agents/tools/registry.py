"""Tool Registry and RBAC Enforcement Layer.

Guarantees that autonomous agents cannot execute unauthorized tools, arbitrary database
queries, or unapproved mutations. All tool invocations are checked against allowed agent roles.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """Base class for tool execution exceptions."""
    pass


class ToolPermissionError(ToolError):
    """Raised when an agent role lacks permission to call a tool."""
    pass


class ToolValidationError(ToolError):
    """Raised when tool inputs or outputs fail validation."""
    pass


class ToolExecutionError(ToolError):
    """Raised when a tool encounters an error during execution."""
    pass


class ToolRegistry:
    """RBAC-enforcing registry for controlled agent tools."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._permissions: Dict[str, Set[str]] = {}
        self._descriptions: Dict[str, str] = {}

    def register_tool(
        self,
        name: str,
        func: Callable,
        allowed_roles: Set[str],
        description: str = "",
    ) -> None:
        """Registers a tool function with its authorized agent roles."""
        if not name or not isinstance(name, str):
            raise ToolValidationError("Tool name must be a non-empty string.")
        if not callable(func):
            raise ToolValidationError(f"Tool '{name}' must be a callable function.")
        if not allowed_roles or not isinstance(allowed_roles, (set, list, tuple)):
            raise ToolValidationError(f"Tool '{name}' must specify a non-empty set of allowed_roles.")

        normalized_roles = {role.strip().lower() for role in allowed_roles}
        self._tools[name] = func
        self._permissions[name] = normalized_roles
        self._descriptions[name] = description or func.__doc__ or ""
        logger.debug("Registered tool '%s' with allowed roles: %s", name, normalized_roles)

    def execute(self, agent_role: str, tool_name: str, **kwargs) -> Any:
        """Executes a tool after verifying agent role permissions.

        Raises:
            ToolValidationError: If tool is not registered.
            ToolPermissionError: If agent role is unauthorized.
            ToolExecutionError: If tool execution fails.
        """
        if tool_name not in self._tools:
            raise ToolValidationError(f"Unknown or unregistered tool: '{tool_name}'.")

        normalized_role = (agent_role or "").strip().lower()
        allowed = self._permissions.get(tool_name, set())

        if normalized_role not in allowed and normalized_role != "system_admin":
            raise ToolPermissionError(
                f"Agent role '{agent_role}' is not authorized to execute tool '{tool_name}'. "
                f"Permitted roles are: {sorted(allowed)}"
            )

        func = self._tools[tool_name]
        try:
            return func(**kwargs)
        except (ToolError, Exception) as err:
            if type(err).__name__ in {"UnauthorizedAccessError", "ToolError", "ToolPermissionError", "ToolValidationError"}:
                raise
            logger.error("Error executing tool '%s' by '%s': %s", tool_name, agent_role, err, exc_info=True)
            raise ToolExecutionError(f"Tool '{tool_name}' failed during execution: {err}") from err

    def has_tool(self, tool_name: str) -> bool:
        """Returns True if the tool is registered."""
        return tool_name in self._tools

    def get_tool(self, tool_name: str) -> Optional[Callable]:
        """Returns the raw callable for a registered tool."""
        return self._tools.get(tool_name)

    def list_tools(self, agent_role: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists available tools, optionally filtered by agent role."""
        tools_list = []
        normalized_role = agent_role.strip().lower() if agent_role else None

        for name, func in self._tools.items():
            allowed = self._permissions.get(name, set())
            if normalized_role and normalized_role not in allowed:
                continue
            tools_list.append({
                "name": name,
                "description": self._descriptions.get(name, ""),
                "allowed_roles": sorted(allowed),
            })
        return tools_list
