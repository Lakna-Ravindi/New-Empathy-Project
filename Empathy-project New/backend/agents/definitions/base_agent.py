"""Abstract BaseAgent interface for Empathy Guru specialized agents."""

from __future__ import annotations

import abc
import logging
from typing import Any, Callable, Dict, List, Optional

from agents.state import AgentState
from agents.tools.registry import ToolPermissionError, ToolRegistry
from agents.tools import get_default_tool_registry

logger = logging.getLogger(__name__)


class AgentExecutionError(Exception):
    """Raised when an agent fails to execute its assigned responsibility."""
    pass


class AgentTimeoutError(AgentExecutionError):
    """Raised when an agent execution exceeds its configured timeout."""
    pass


class BaseAgent(abc.ABC):
    """Abstract base class for all specialized domain agents.

    Enforces:
    - Structured inputs and outputs.
    - Controlled tool execution with RBAC enforcement via ToolRegistry.
    - Timeout and graceful error handling.
    """

    def __init__(
        self,
        role: str,
        tool_registry: Optional[ToolRegistry] = None,
        timeout_seconds: float = 15.0,
    ):
        self.role = (role or "").strip().lower()
        self.tool_registry = tool_registry if tool_registry is not None else get_default_tool_registry()
        self.timeout_seconds = timeout_seconds

    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """Invokes a controlled tool through the RBAC registry.

        Raises:
            ToolPermissionError: If this agent role is unauthorized to call the tool.
            ToolExecutionError: If tool execution encounters an error.
        """
        logger.debug("Agent '%s' invoking tool '%s' with args %s", self.role, tool_name, list(kwargs.keys()))
        return self.tool_registry.execute(agent_role=self.role, tool_name=tool_name, **kwargs)

    @abc.abstractmethod
    def execute(self, input_data: Any, state: Optional[AgentState] = None) -> Any:
        """Executes the agent's core responsibility with structured input and output."""
        raise NotImplementedError
