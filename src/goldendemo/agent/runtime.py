"""Runtime layer for OpenAI Responses API interaction and tool dispatch."""

import json
import logging
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from openai import OpenAI
from openai.types.responses import FunctionToolParam

from goldendemo.agent.errors import FatalToolError, is_fatal_error
from goldendemo.agent.events import (
    AgentEvent,
    error_event,
    guardrail_warning_event,
    reasoning_event,
    tool_call_event,
    tool_result_event,
)

if TYPE_CHECKING:
    from goldendemo.agent.state import AgentState
    from goldendemo.agent.tools import BaseTool

logger = logging.getLogger(__name__)


@dataclass
class ToolCallResult:
    """Result from a single tool call."""

    tool_name: str
    call_id: str
    args: dict[str, Any]
    result: Any  # ToolResult object
    success: bool


@dataclass
class ProcessedResult:
    """Result from processing a model response."""

    function_calls: list[Any] = field(default_factory=list)
    tool_results: list[ToolCallResult] = field(default_factory=list)
    error: bool = False


class ResponseRunner:
    """Handles OpenAI Responses API interaction and tool dispatch.

    This class encapsulates the low-level mechanics of:
    - Calling the OpenAI Responses API
    - Parsing response items (reasoning, messages, function calls)
    - Executing tools and handling errors
    - Managing context accumulation
    """

    def __init__(
        self,
        openai_client: OpenAI,
        model: str,
        tools: dict[str, "BaseTool"],
        responses_tools: list[FunctionToolParam],
        reasoning_effort: str = "medium",
        reasoning_summary: bool = True,
    ):
        """Initialize the runtime.

        Args:
            openai_client: OpenAI client instance.
            model: Model name to use.
            tools: Dictionary of tool name -> BaseTool instances.
            responses_tools: List of FunctionToolParam for the API.
            reasoning_effort: Reasoning effort level.
            reasoning_summary: Whether to include reasoning summaries.
        """
        self.openai_client = openai_client
        self.model = model
        self.tools = tools
        self.responses_tools = responses_tools
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary

    def call_model(
        self,
        state: "AgentState",
        input_items: list[dict[str, Any]],
        guardrails: dict[str, Any] | None = None,
    ) -> Generator[AgentEvent, None, Any]:
        """Call the model for one iteration.

        Args:
            state: Current agent state (for token tracking).
            input_items: Conversation context.
            guardrails: Optional guardrails dict to check iteration budget.

        Yields:
            AgentEvent objects for warnings/errors.

        Returns:
            The response object or None if an error occurred.
        """
        # Check iteration budget guardrail if provided
        if guardrails and "iteration_budget" in guardrails:
            budget_check = guardrails["iteration_budget"].check(state)
            if budget_check.warning:
                yield guardrail_warning_event("iteration_budget", budget_check.warning)
                input_items.append(
                    {
                        "type": "message",
                        "role": "system",
                        "content": f"WARNING: {budget_check.warning}",
                    }
                )

        try:
            # Build reasoning config
            reasoning_config: dict[str, Any] = {"effort": self.reasoning_effort}
            include_params: list[str] = []

            if self.reasoning_summary:
                reasoning_config["summary"] = "auto"
                include_params.append("reasoning.encrypted_content")

            # Note: No max_output_tokens - truncation wastes money (pay for truncated + retry)
            response = self.openai_client.responses.create(
                model=self.model,
                input=input_items,
                tools=self.responses_tools,
                tool_choice="required",  # model must call a tool every turn
                reasoning=reasoning_config,
                include=include_params if include_params else None,
            )

            # Track token usage
            if hasattr(response, "usage") and response.usage:
                usage_dict = response.usage.model_dump() if hasattr(response.usage, "model_dump") else response.usage
                state.token_usage.add_usage(usage_dict)
                logger.debug(
                    f"Token usage this call: {usage_dict.get('total_tokens', 0)} "
                    f"(input: {usage_dict.get('input_tokens', 0)}, "
                    f"output: {usage_dict.get('output_tokens', 0)})"
                )

            return response

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Model call failed: {e}")

            if "verified to generate reasoning summaries" in error_msg:
                yield error_event(
                    "Reasoning summaries require a verified OpenAI organization.\n"
                    "Options:\n"
                    "  1. Verify your org: https://platform.openai.com/settings/organization/general\n"
                    "  2. Run with --no-reasoning-summary flag to disable summaries",
                    recoverable=False,
                )
            else:
                yield error_event(error_msg, recoverable=False)
            return None

    def process_response(
        self,
        state: "AgentState",
        response: Any,
        input_items: list[dict[str, Any]],
        accumulate_context: bool = True,
    ) -> Generator[AgentEvent, None, ProcessedResult]:
        """Process model response and execute tool calls.

        Args:
            state: Current agent state.
            response: Model response object.
            input_items: Current conversation context.
            accumulate_context: Whether to add results to input_items.

        Yields:
            AgentEvent objects for tool calls/results.

        Returns:
            ProcessedResult with function calls and status flags.
        """
        # Check for incomplete response
        if response.status == "incomplete":
            reason = getattr(response.incomplete_details, "reason", "unknown")
            logger.warning(f"Response incomplete: {reason}")

        function_calls = []

        for item in response.output:
            # Handle reasoning items
            if item.type == "reasoning":
                if item.summary:
                    summary_texts = [s.text for s in item.summary if hasattr(s, "text") and s.text]
                    if summary_texts:
                        yield reasoning_event(" ".join(summary_texts), item.id)

                if accumulate_context:
                    input_items.append(
                        {
                            "type": "reasoning",
                            "id": item.id,
                            "summary": [
                                {"type": s.type, "text": s.text}
                                for s in (item.summary or [])
                                if hasattr(s, "type") and hasattr(s, "text")
                            ],
                            "encrypted_content": getattr(item, "encrypted_content", None),
                        }
                    )

            # Handle message items
            elif item.type == "message":
                content = item.content[0].text if item.content else ""
                if content and accumulate_context:
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": content,
                        }
                    )

            # Handle function calls
            elif item.type == "function_call":
                function_calls.append(item)
                if accumulate_context:
                    input_items.append(
                        {
                            "type": "function_call",
                            "id": item.id,
                            "call_id": item.call_id,
                            "name": item.name,
                            "arguments": item.arguments,
                        }
                    )

        # Execute function calls and collect results
        tool_results: list[ToolCallResult] = []
        for func_call in function_calls:
            tool_result, should_stop = yield from self._execute_function_call(
                state, func_call, input_items, accumulate_context
            )
            if tool_result:
                tool_results.append(tool_result)
            if should_stop:
                return ProcessedResult(
                    function_calls=function_calls,
                    tool_results=tool_results,
                    error=True,
                )

        return ProcessedResult(
            function_calls=function_calls,
            tool_results=tool_results,
            error=False,
        )

    def _execute_function_call(
        self,
        state: "AgentState",
        func_call: Any,
        input_items: list[dict[str, Any]],
        accumulate_context: bool,
    ) -> Generator[AgentEvent, None, tuple[ToolCallResult | None, bool]]:
        """Execute a single function call.

        Returns:
            Tuple of (ToolCallResult or None, should_stop).
            should_stop is True if a fatal error occurred.
        """
        tool_name = func_call.name
        call_id = func_call.call_id

        # Parse and validate arguments against tool schema
        tool_args, parse_error = self._parse_tool_args(func_call.arguments, tool_name)

        # Handle parse/validation failure
        if tool_args is None:
            error_output: dict[str, Any] = {"success": False, "data": None, "error": parse_error, "metadata": {}}
            if accumulate_context:
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(error_output),
                    }
                )
                input_items.append(
                    {
                        "type": "message",
                        "role": "system",
                        "content": f"CORRECTION: Tool call to {tool_name} failed validation: {parse_error}. "
                        f"Check required fields and argument types.",
                    }
                )
            yield tool_result_event(tool_name, error_output, call_id, success=False)
            return (None, False)  # Continue to next function call

        yield tool_call_event(tool_name, tool_args, call_id)

        # Execute tool
        try:
            result = self._execute_tool(tool_name, state, tool_args)
        except FatalToolError as e:
            logger.error(f"Fatal tool error, aborting: {e}")
            yield error_event(str(e), recoverable=False)
            return (None, True)  # Stop - fatal error

        yield tool_result_event(tool_name, result.to_dict(), call_id, result.success)

        if accumulate_context:
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result.to_dict()),
                }
            )

        tool_call_result = ToolCallResult(
            tool_name=tool_name,
            call_id=call_id,
            args=tool_args,
            result=result,
            success=result.success,
        )
        return (tool_call_result, False)  # Continue processing

    def _parse_tool_args(self, raw_args: Any, tool_name: str | None = None) -> tuple[dict | None, str | None]:
        """Parse and validate tool arguments.

        Returns:
            Tuple of (parsed_args, error_message). One will be None.
        """
        # Parse JSON if needed
        if isinstance(raw_args, dict):
            parsed = raw_args
        elif isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool args: {e}")
                logger.debug(f"Raw arguments: {raw_args[:500] if raw_args else 'empty'}")
                return None, f"Invalid JSON in tool arguments: {e}"
        else:
            logger.warning(f"Unexpected argument type: {type(raw_args)}")
            return None, f"Unexpected argument type: {type(raw_args)}"

        # Validate against tool schema if tool_name provided
        if tool_name and tool_name in self.tools:
            validation_error = self._validate_tool_args(tool_name, parsed)
            if validation_error:
                return None, validation_error

        return parsed, None

    def _validate_tool_args(self, tool_name: str, args: dict) -> str | None:
        """Validate arguments against tool's JSON schema.

        Args:
            tool_name: Name of the tool.
            args: Parsed arguments dict.

        Returns:
            Error message if validation fails, None if valid.
        """
        tool = self.tools[tool_name]
        schema = tool.parameters

        # Check required fields
        required = schema.get("required", [])
        missing = [field for field in required if field not in args]
        if missing:
            return f"Missing required field(s): {', '.join(missing)}"

        # Validate each provided field against schema
        properties = schema.get("properties", {})
        errors = []

        for field_name, value in args.items():
            if value is None:
                continue  # None values are cleaned later

            if field_name not in properties:
                continue  # Extra fields are allowed (cleaned later)

            field_schema = properties[field_name]
            field_type = field_schema.get("type")

            # Type validation
            type_error = self._check_field_type(field_name, value, field_type)
            if type_error:
                errors.append(type_error)
                continue

            # Range validation for numbers
            if field_type in ("integer", "number"):
                if "minimum" in field_schema and value < field_schema["minimum"]:
                    errors.append(f"Field '{field_name}' value {value} is below minimum {field_schema['minimum']}")
                if "maximum" in field_schema and value > field_schema["maximum"]:
                    errors.append(f"Field '{field_name}' value {value} exceeds maximum {field_schema['maximum']}")

        if errors:
            return "; ".join(errors)

        return None

    def _check_field_type(self, field: str, value: Any, expected_type: str | None) -> str | None:
        """Check if a value matches the expected JSON schema type.

        Returns:
            Error message if type mismatch, None if valid.
        """
        if expected_type is None:
            return None

        type_checks = {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
        }

        check = type_checks.get(expected_type)
        if check and not check(value):
            return f"Field '{field}' expected {expected_type}, got {type(value).__name__}"

        return None

    def _execute_tool(self, tool_name: str, state: "AgentState", args: dict[str, Any]) -> Any:
        """Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute.
            state: Current agent state.
            args: Tool arguments.

        Returns:
            ToolResult from the tool execution.

        Raises:
            FatalToolError: If a fatal infrastructure error occurs.
        """
        from goldendemo.agent.tools.base import ToolResult

        tool = self.tools.get(tool_name)
        if not tool:
            return ToolResult.fail(f"Unknown tool: {tool_name}")

        # Clean args - remove null/None values that model sometimes sends
        cleaned_args = {k: v for k, v in args.items() if v is not None and k != "null"}

        try:
            return tool.execute(state, **cleaned_args)
        except Exception as e:
            # Classify error as fatal or recoverable
            if is_fatal_error(e):
                logger.error(f"Fatal error in tool {tool_name}: {e}")
                raise FatalToolError(f"Tool {tool_name} failed with infrastructure error: {e}") from e

            # Recoverable error - return as tool result so agent can adjust
            logger.warning(f"Recoverable error in tool {tool_name}: {e}")
            return ToolResult.fail(f"Tool error (recoverable): {e}")

    def handle_no_tool_call(
        self,
        state: "AgentState",
        response: Any,
        input_items: list[dict[str, Any]],
    ) -> Generator[AgentEvent, None, None]:
        """Handle case where model returns no function calls.

        Args:
            state: Current agent state.
            response: Model response.
            input_items: Conversation context to append correction to.

        Yields:
            Warning event.
        """
        assistant_msg = ""
        for item in response.output:
            if item.type == "message" and item.content:
                assistant_msg = item.content[0].text if item.content else ""
                break

        logger.warning(f"Model returned no function calls: {assistant_msg[:200] if assistant_msg else 'no message'}")

        feedback = "You returned text instead of a tool call. You MUST call a tool every turn. "
        if state.plan_submitted:
            step = state.get_current_step()
            if step:
                feedback += f'Call browse_category(product_class="{step.category}").'
            else:
                feedback += "All steps complete - call finish_judgments() to finalize."
        else:
            feedback += "Call list_categories() or search_products() to explore the catalog."

        state.add_guardrail_feedback(feedback)

        # Inject feedback directly into input_items so it's seen on next iteration
        input_items.append(
            {
                "type": "message",
                "role": "system",
                "content": f"CORRECTION: {feedback}",
            }
        )

        yield guardrail_warning_event("no_tool_call", "Agent returned text - reminded to use tools")
