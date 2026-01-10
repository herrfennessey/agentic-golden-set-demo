"""Base classes for agent guardrails."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from goldendemo.agent.state import AgentState


class GuardrailAction(str, Enum):
    """Action to take when guardrail check fails or warns."""

    BLOCK = "block"  # Block submission, require fix
    WARN = "warn"  # Allow but flag for review
    SUGGEST = "suggest"  # Soft suggestion, no blocking


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    passed: bool
    message: str | None = None
    warning: str | None = None
    action: GuardrailAction = GuardrailAction.BLOCK
    review_questions: list[str] = field(default_factory=list)

    @classmethod
    def success(cls) -> "GuardrailResult":
        """Create a successful result."""
        return cls(passed=True)

    @classmethod
    def failure(cls, message: str, action: GuardrailAction = GuardrailAction.BLOCK) -> "GuardrailResult":
        """Create a failure result."""
        return cls(passed=False, message=message, action=action)

    @classmethod
    def with_warning(cls, warning: str, review_questions: list[str] | None = None) -> "GuardrailResult":
        """Create a passing result with a warning."""
        return cls(
            passed=True,
            warning=warning,
            action=GuardrailAction.WARN,
            review_questions=review_questions or [],
        )


class Guardrail(ABC):
    """Abstract base class for guardrails.

    Guardrails validate agent state and submissions to ensure quality
    and prevent pathological outputs.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this guardrail."""
        ...

    @abstractmethod
    def check(self, state: "AgentState", **kwargs: Any) -> GuardrailResult:
        """Run the guardrail check.

        Args:
            state: Current agent state.
            **kwargs: Additional context-specific arguments.

        Returns:
            GuardrailResult indicating pass/fail and any messages.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
