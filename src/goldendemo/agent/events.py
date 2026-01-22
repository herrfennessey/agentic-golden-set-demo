"""Streaming events for agent UI updates."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Types of events emitted during agent execution."""

    STARTED = "started"
    ITERATION_START = "iteration_start"
    EXECUTION_PHASE_STARTED = "execution_phase_started"
    REASONING = "reasoning"  # Reasoning summaries from OpenAI Responses API
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    GUARDRAIL_WARNING = "guardrail_warning"
    GUARDRAIL_FAILURE = "guardrail_failure"
    JUDGMENT_ADDED = "judgment_added"
    PLAN_STEP_COMPLETED = "plan_step_completed"
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


def reasoning_event(summary: str, reasoning_id: str | None = None) -> AgentEvent:
    """Create a reasoning event from OpenAI Responses API reasoning summaries."""
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
    token_usage: dict[str, int] | None = None,
) -> AgentEvent:
    """Create a completed event."""
    data = {
        "status": status,
        "judgments_count": judgments_count,
        "tool_calls": tool_calls,
        "warnings": warnings or [],
    }
    if token_usage:
        data["token_usage"] = token_usage
    return AgentEvent(
        type=EventType.COMPLETED,
        data=data,
    )


def error_event(error: str, recoverable: bool = False) -> AgentEvent:
    """Create an error event."""
    return AgentEvent(
        type=EventType.ERROR,
        data={"error": error, "recoverable": recoverable},
    )


def execution_phase_started_event(plan_steps: int) -> AgentEvent:
    """Create an event indicating the execution phase has started.

    Emitted after the discovery phase completes and the agent has submitted a plan.
    """
    return AgentEvent(
        type=EventType.EXECUTION_PHASE_STARTED,
        data={"plan_steps": plan_steps},
    )


def plan_step_completed_event(
    step_index: int,
    category: str,
    summary: str,
    has_more_steps: bool,
) -> AgentEvent:
    """Create an event indicating a plan step (category browse) completed."""
    return AgentEvent(
        type=EventType.PLAN_STEP_COMPLETED,
        data={
            "step_index": step_index,
            "category": category,
            "summary": summary,
            "has_more_steps": has_more_steps,
        },
    )


def search_step_event(
    step_index: int,
    query: str,
    products_found: int,
    judgments_added: int,
    exact_count: int,
    partial_count: int,
) -> AgentEvent:
    """Create an event indicating a search step executed and auto-completed."""
    return AgentEvent(
        type=EventType.PLAN_STEP_COMPLETED,
        data={
            "step_index": step_index,
            "step_type": "search",
            "query": query,
            "products_found": products_found,
            "judgments_added": judgments_added,
            "exact_count": exact_count,
            "partial_count": partial_count,
        },
    )
