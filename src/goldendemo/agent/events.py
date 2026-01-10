"""Streaming events for agent UI updates."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Types of events emitted during agent execution."""

    STARTED = "started"
    ITERATION_START = "iteration_start"
    PHASE_CHANGE = "phase_change"
    THINKING = "thinking"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    GUARDRAIL_WARNING = "guardrail_warning"
    GUARDRAIL_FAILURE = "guardrail_failure"
    JUDGMENT_ADDED = "judgment_added"
    STEP_COMPLETED = "step_completed"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentEvent:
    """Base event emitted during agent execution."""

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


def started_event(query: str, query_id: str | None = None) -> AgentEvent:
    """Create a started event."""
    return AgentEvent(
        type=EventType.STARTED,
        data={"query": query, "query_id": query_id},
    )


def iteration_start_event(iteration: int, max_iterations: int) -> AgentEvent:
    """Create an iteration start event."""
    return AgentEvent(
        type=EventType.ITERATION_START,
        data={"iteration": iteration, "max_iterations": max_iterations},
    )


def thinking_event(content: str) -> AgentEvent:
    """Create a thinking event (model's internal reasoning)."""
    return AgentEvent(
        type=EventType.THINKING,
        data={"content": content},
    )


def reasoning_event(summary: str, reasoning_id: str | None = None) -> AgentEvent:
    """Create a reasoning event (from Responses API)."""
    return AgentEvent(
        type=EventType.REASONING,
        data={"summary": summary, "reasoning_id": reasoning_id},
    )


def tool_call_event(
    tool_name: str,
    arguments: dict[str, Any],
    call_id: str | None = None,
) -> AgentEvent:
    """Create a tool call event."""
    return AgentEvent(
        type=EventType.TOOL_CALL,
        data={
            "tool_name": tool_name,
            "arguments": arguments,
            "call_id": call_id,
        },
    )


def tool_result_event(
    tool_name: str,
    result: dict[str, Any],
    call_id: str | None = None,
    success: bool = True,
) -> AgentEvent:
    """Create a tool result event."""
    return AgentEvent(
        type=EventType.TOOL_RESULT,
        data={
            "tool_name": tool_name,
            "result": result,
            "call_id": call_id,
            "success": success,
        },
    )


def guardrail_warning_event(guardrail_name: str, warning: str) -> AgentEvent:
    """Create a guardrail warning event."""
    return AgentEvent(
        type=EventType.GUARDRAIL_WARNING,
        data={"guardrail": guardrail_name, "warning": warning},
    )


def guardrail_failure_event(guardrail_name: str, message: str) -> AgentEvent:
    """Create a guardrail failure event."""
    return AgentEvent(
        type=EventType.GUARDRAIL_FAILURE,
        data={"guardrail": guardrail_name, "message": message},
    )


def judgment_added_event(
    product_id: str,
    relevance: int,
    reasoning: str,
) -> AgentEvent:
    """Create a judgment added event."""
    return AgentEvent(
        type=EventType.JUDGMENT_ADDED,
        data={
            "product_id": product_id,
            "relevance": relevance,
            "reasoning": reasoning[:100],  # Truncate for event
        },
    )


def completed_event(
    status: str,
    judgments_count: int,
    tool_calls: int,
    warnings: list[str] | None = None,
) -> AgentEvent:
    """Create a completed event."""
    return AgentEvent(
        type=EventType.COMPLETED,
        data={
            "status": status,
            "judgments_count": judgments_count,
            "tool_calls": tool_calls,
            "warnings": warnings or [],
        },
    )


def error_event(error: str, recoverable: bool = False) -> AgentEvent:
    """Create an error event."""
    return AgentEvent(
        type=EventType.ERROR,
        data={"error": error, "recoverable": recoverable},
    )


def phase_change_event(
    from_phase: str,
    to_phase: str,
    plan_steps: int | None = None,
) -> AgentEvent:
    """Create a phase change event."""
    return AgentEvent(
        type=EventType.PHASE_CHANGE,
        data={
            "from_phase": from_phase,
            "to_phase": to_phase,
            "plan_steps": plan_steps,
        },
    )


def step_completed_event(
    step_index: int,
    category: str,
    summary: str,
    has_more_steps: bool,
) -> AgentEvent:
    """Create a step completed event."""
    return AgentEvent(
        type=EventType.STEP_COMPLETED,
        data={
            "step_index": step_index,
            "category": category,
            "summary": summary,
            "has_more_steps": has_more_steps,
        },
    )
