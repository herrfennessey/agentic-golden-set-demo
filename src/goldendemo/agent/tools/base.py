"""Base classes for agent tools."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from goldendemo.agent.state import AgentState
    from goldendemo.clients.weaviate_client import WeaviateClient


@dataclass
class ToolResult:
    """Result of a tool execution."""

    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, data: Any, **metadata: Any) -> "ToolResult":
        """Create a successful result."""
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def fail(cls, error: str) -> "ToolResult":
        """Create a failure result."""
        return cls(success=False, error=error)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Always returns a consistent shape: {success, data, error, metadata}
        """
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }


class BaseTool(ABC):
    """Abstract base class for agent tools.

    Tools provide the agent with capabilities to explore the product
    catalog and submit judgments.
    """

    def __init__(self, weaviate_client: "WeaviateClient"):
        """Initialize with Weaviate client.

        Args:
            weaviate_client: Connected Weaviate client for product queries.
        """
        self.weaviate_client = weaviate_client

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for function calling."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for the LLM."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        ...

    @abstractmethod
    def execute(self, state: "AgentState", **kwargs: Any) -> ToolResult:
        """Execute the tool.

        Args:
            state: Current agent state (may be updated by tool).
            **kwargs: Tool-specific arguments.

        Returns:
            ToolResult with success/failure and data.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
