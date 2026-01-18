"""Tests for agent streaming events."""

from goldendemo.agent.events import (
    AgentEvent,
    EventType,
    completed_event,
    error_event,
    guardrail_failure_event,
    guardrail_warning_event,
    iteration_start_event,
    judgment_added_event,
    reasoning_event,
    started_event,
    tool_call_event,
    tool_result_event,
)


class TestEventType:
    """Tests for EventType enum."""

    def test_all_event_types_exist(self):
        """Test that all expected event types are defined."""
        expected = [
            "STARTED",
            "ITERATION_START",
            "EXECUTION_PHASE_STARTED",
            "REASONING",
            "TOOL_CALL",
            "TOOL_RESULT",
            "GUARDRAIL_WARNING",
            "GUARDRAIL_FAILURE",
            "JUDGMENT_ADDED",
            "PLAN_STEP_COMPLETED",
            "COMPLETED",
            "ERROR",
        ]

        for event_type in expected:
            assert hasattr(EventType, event_type)


class TestAgentEvent:
    """Tests for AgentEvent base class."""

    def test_creation(self):
        """Test event creation."""
        event = AgentEvent(type=EventType.STARTED, data={"query": "test"})

        assert event.type == EventType.STARTED
        assert event.data["query"] == "test"
        assert event.timestamp is not None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        event = AgentEvent(type=EventType.STARTED, data={"query": "test"})

        result = event.to_dict()

        assert result["type"] == "started"
        assert result["data"]["query"] == "test"
        assert "timestamp" in result


class TestEventFactories:
    """Tests for event factory functions."""

    def test_started_event(self):
        """Test started event creation."""
        event = started_event("blue sofa", "q123")

        assert event.type == EventType.STARTED
        assert event.data["query"] == "blue sofa"
        assert event.data["query_id"] == "q123"

    def test_iteration_start_event(self):
        """Test iteration start event creation."""
        event = iteration_start_event(5, 20)

        assert event.type == EventType.ITERATION_START
        assert event.data["iteration"] == 5
        assert event.data["max_iterations"] == 20

    def test_reasoning_event(self):
        """Test reasoning event creation."""
        event = reasoning_event("This query is about furniture", "r123")

        assert event.type == EventType.REASONING
        assert event.data["summary"] == "This query is about furniture"
        assert event.data["reasoning_id"] == "r123"

    def test_tool_call_event(self):
        """Test tool call event creation."""
        event = tool_call_event(
            "search_products",
            {"query": "sofa", "limit": 10},
            "call_123",
        )

        assert event.type == EventType.TOOL_CALL
        assert event.data["tool_name"] == "search_products"
        assert event.data["arguments"]["query"] == "sofa"
        assert event.data["call_id"] == "call_123"

    def test_tool_result_event(self):
        """Test tool result event creation."""
        event = tool_result_event(
            "search_products",
            {"success": True, "count": 25},
            "call_123",
            success=True,
        )

        assert event.type == EventType.TOOL_RESULT
        assert event.data["tool_name"] == "search_products"
        assert event.data["result"]["count"] == 25
        assert event.data["success"] is True

    def test_guardrail_warning_event(self):
        """Test guardrail warning event creation."""
        event = guardrail_warning_event("iteration_budget", "Running low on iterations")

        assert event.type == EventType.GUARDRAIL_WARNING
        assert event.data["guardrail"] == "iteration_budget"
        assert event.data["warning"] == "Running low on iterations"

    def test_guardrail_failure_event(self):
        """Test guardrail failure event creation."""
        event = guardrail_failure_event("minimum_exploration", "Not enough exploration")

        assert event.type == EventType.GUARDRAIL_FAILURE
        assert event.data["guardrail"] == "minimum_exploration"
        assert event.data["message"] == "Not enough exploration"

    def test_judgment_added_event(self):
        """Test judgment added event creation."""
        event = judgment_added_event("p123", 2, "This product matches exactly")

        assert event.type == EventType.JUDGMENT_ADDED
        assert event.data["product_id"] == "p123"
        assert event.data["relevance"] == 2
        assert "matches" in event.data["reasoning"]

    def test_judgment_added_event_truncates_reasoning(self):
        """Test that long reasoning is truncated in event."""
        long_reasoning = "x" * 200
        event = judgment_added_event("p123", 2, long_reasoning)

        assert len(event.data["reasoning"]) == 100

    def test_completed_event(self):
        """Test completed event creation."""
        event = completed_event(
            status="success",
            judgments_count=25,
            tool_calls=8,
            warnings=["Some warning"],
        )

        assert event.type == EventType.COMPLETED
        assert event.data["status"] == "success"
        assert event.data["judgments_count"] == 25
        assert event.data["tool_calls"] == 8
        assert event.data["warnings"] == ["Some warning"]

    def test_completed_event_default_warnings(self):
        """Test completed event with no warnings."""
        event = completed_event("success", 10, 5)

        assert event.data["warnings"] == []

    def test_error_event(self):
        """Test error event creation."""
        event = error_event("Something went wrong", recoverable=True)

        assert event.type == EventType.ERROR
        assert event.data["error"] == "Something went wrong"
        assert event.data["recoverable"] is True

    def test_error_event_default_recoverable(self):
        """Test error event default recoverable value."""
        event = error_event("Error")

        assert event.data["recoverable"] is False
