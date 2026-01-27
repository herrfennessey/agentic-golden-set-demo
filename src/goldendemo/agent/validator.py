"""Validation subagent for reviewing judgments before save.

This subagent reviews all judgments with fresh context to catch hallucinated
products or incorrect relevance scores before the golden set is saved.
"""

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from goldendemo.agent.state import AgentState

from openai import OpenAI

from goldendemo.config import settings

logger = logging.getLogger(__name__)


# Validation prompt for subagent
VALIDATION_PROMPT = """You are a validation reviewer for product relevance judgments in e-commerce search.

Query: "{query}"

## Your Job
Review each judgment below and verify the relevance score is CORRECT. You can:
- KEEP correct judgments as-is
- ADJUST relevance scores (upgrade Partial→Exact or downgrade Exact→Partial)
- REMOVE products that don't belong at all

## Relevance Levels
- Exact (2): Product DIRECTLY matches the query - exactly what the user wants
- Partial (1): Product is RELATED but not a direct match (wrong size, color, style, etc.)

## What to Check

### 1. Size/Measurement Mismatches (CRITICAL)
- If the query specifies a size (e.g., "7 qt", "5 ft", "queen size"), verify the product matches
- Wrong size = Partial, not Exact
- Example: "7 qt slow cooker" query → 6 qt cooker should be Partial (adjust if marked Exact)

### 2. Verify Relevance Score Accuracy
- Exact (2): Does this product DIRECTLY match what the user searched for?
- Partial (1): Is this related but not exactly what they want?
- If a Partial product actually matches perfectly → adjust to Exact
- If an Exact product has wrong attributes → adjust to Partial

### 3. Products That Don't Belong
- Completely unrelated products should be REMOVED
- Example: "slow cooker" query → a "Blender" product should be removed

## Judgments to Review

{judgments_list}

## Response Format

Call submit_validation with three arrays:
- `keep`: Product IDs that are CORRECT and should stay unchanged
- `adjust`: Objects with product_id, new_relevance (1 or 2), and reason for score changes
- `remove`: Objects with product_id and reason for products to REMOVE entirely

Be thoughtful: adjust scores when needed, but only remove products that truly don't belong."""


def _format_judgments_for_validation(judgments_with_products: list[dict]) -> str:
    """Format judgments with product data for the validation prompt."""
    lines = []
    for i, item in enumerate(judgments_with_products, 1):
        j = item["judgment"]
        p = item["product"]

        relevance_label = "Exact" if j["relevance"] == 2 else "Partial"

        lines.append(f"--- Judgment #{i} ---")
        lines.append(f"PRODUCT_ID: {j['product_id']}")
        lines.append(f"Relevance: {relevance_label} ({j['relevance']})")
        lines.append(f"Judge's Reasoning: {j.get('reasoning', 'N/A')}")
        lines.append("")
        lines.append("Product Data:")
        lines.append(f"  Name: {p.get('product_name', 'Unknown')}")
        lines.append(f"  Category: {p.get('product_class', 'Unknown')}")
        if p.get("product_description"):
            # Truncate long descriptions
            desc = p["product_description"][:300]
            if len(p["product_description"]) > 300:
                desc += "..."
            lines.append(f"  Description: {desc}")
        if p.get("product_features"):
            lines.append(f"  Features: {p['product_features'][:200]}")
        lines.append("")
    return "\n".join(lines)


class ValidationSubagent:
    """Subagent for validating judgments before saving the golden set.

    This subagent reviews all judgments with fresh context to catch:
    - Hallucinated products (IDs that don't match product data)
    - Incorrect relevance scores (wrong size, category mismatch, etc.)
    - Fabricated or suspicious product data
    """

    def __init__(
        self,
        openai_client: OpenAI | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ):
        """Initialize the validation subagent.

        Args:
            openai_client: OpenAI client. Creates one if not provided.
            model: Model to use (defaults to VALIDATE_MODEL from .env).
            reasoning_effort: Reasoning effort level (defaults to VALIDATE_REASONING_EFFORT from .env).
        """
        self.openai_client = openai_client or OpenAI(api_key=settings.openai_api_key)
        self.model = model if model is not None else settings.validate_model
        self.reasoning_effort = reasoning_effort if reasoning_effort is not None else settings.validate_reasoning_effort

        # Define the submit_validation tool for structured output
        self.tools = [
            {
                "type": "function",
                "name": "submit_validation",
                "description": "Submit validation results for reviewed judgments.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keep": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Product IDs that are correct and should be kept unchanged",
                        },
                        "adjust": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "product_id": {
                                        "type": "string",
                                        "description": "Product ID to adjust",
                                    },
                                    "new_relevance": {
                                        "type": "integer",
                                        "enum": [1, 2],
                                        "description": "New relevance score: 1=Partial, 2=Exact",
                                    },
                                    "reason": {
                                        "type": "string",
                                        "description": "Why this relevance score should be changed",
                                    },
                                },
                                "required": ["product_id", "new_relevance", "reason"],
                                "additionalProperties": False,
                            },
                            "description": "Products to adjust relevance score (upgrade or downgrade)",
                        },
                        "remove": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "product_id": {
                                        "type": "string",
                                        "description": "Product ID to remove",
                                    },
                                    "reason": {
                                        "type": "string",
                                        "description": "Why this product should be removed entirely",
                                    },
                                },
                                "required": ["product_id", "reason"],
                                "additionalProperties": False,
                            },
                            "description": "Products to remove entirely (don't belong in golden set)",
                        },
                    },
                    "required": ["keep", "adjust", "remove"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]

    def validate_judgments(
        self,
        query: str,
        judgments_with_products: list[dict],
        state: "AgentState | None" = None,
    ) -> dict[str, Any]:
        """Validate a batch of judgments against their product data.

        Args:
            query: Search query.
            judgments_with_products: List of dicts with "judgment" and "product" keys.
            state: Optional agent state for token tracking.

        Returns:
            Dict with:
            - "keep": list of product_ids to keep unchanged
            - "adjust": list of {product_id, new_relevance, reason} for score changes
            - "remove": list of {product_id, reason} for removals

        Raises:
            RuntimeError: If validation fails after retries.
        """
        if not judgments_with_products:
            logger.warning("validate_judgments called with empty list")
            return {"keep": [], "adjust": [], "remove": []}

        logger.info(f"ValidationSubagent reviewing {len(judgments_with_products)} judgments for query '{query}'")

        # Retry if model doesn't call tool or payload is invalid
        max_attempts = settings.validate_max_retries
        for attempt in range(1, max_attempts + 1):
            try:
                tool_call_valid, result, response_text = self._call_and_extract_validation(
                    query, judgments_with_products, attempt, state
                )

                if tool_call_valid:
                    keep_count = len(result.get("keep", []))
                    adjust_count = len(result.get("adjust", []))
                    remove_count = len(result.get("remove", []))
                    logger.info(
                        f"ValidationSubagent completed on attempt {attempt}: "
                        f"keeping {keep_count}, adjusting {adjust_count}, removing {remove_count}"
                    )
                    return result
                else:
                    logger.warning(f"Attempt {attempt}/{max_attempts}: Validation tool call invalid or missing")
                    logger.warning(f"Response: {response_text[:200]}")
                    if attempt < max_attempts:
                        logger.info("Retrying validation...")
                        continue
                    else:
                        raise RuntimeError(
                            f"ValidationSubagent failed to call submit_validation tool after {max_attempts} attempts."
                        )

            except Exception as e:
                if attempt < max_attempts:
                    logger.warning(f"Validation attempt {attempt} failed: {e}. Retrying...")
                    continue
                else:
                    logger.error(f"ValidationSubagent failed after {max_attempts} attempts: {e}", exc_info=True)
                    raise

        raise RuntimeError(f"ValidationSubagent failed after {max_attempts} attempts")

    def _call_and_extract_validation(
        self,
        query: str,
        judgments_with_products: list[dict],
        attempt: int,
        state: "AgentState | None" = None,
    ) -> tuple[bool, dict[str, Any], str]:
        """Make API call and extract validation results.

        Args:
            query: Search query.
            judgments_with_products: Judgments with product data to validate.
            attempt: Attempt number (for logging).
            state: Optional agent state for token tracking.

        Returns:
            Tuple of (found_tool_call, validation_result, actual_response_text).
        """
        # Format judgments for prompt
        judgments_list = _format_judgments_for_validation(judgments_with_products)

        prompt = VALIDATION_PROMPT.format(
            query=query,
            judgments_list=judgments_list,
        )

        logger.debug(f"Validation attempt {attempt}: Calling OpenAI with model={self.model}")

        response = self.openai_client.responses.create(
            model=self.model,
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": prompt,
                }
            ],
            tools=self.tools,
            tool_choice={"type": "function", "name": "submit_validation"},
            reasoning={"effort": self.reasoning_effort},
        )

        # Check response status
        response_status = getattr(response, "status", "unknown")
        is_incomplete = response_status == "incomplete"

        if is_incomplete:
            logger.warning("Validation response is 'incomplete' - was truncated")

        # Track token usage if state provided
        if state and hasattr(response, "usage") and response.usage:
            usage_dict = response.usage.model_dump() if hasattr(response.usage, "model_dump") else response.usage
            state.token_usage.add_usage(usage_dict)
            logger.debug(
                f"Validation tokens: input={usage_dict.get('input_tokens', 0)}, "
                f"output={usage_dict.get('output_tokens', 0)}"
            )

        # Extract validation results from tool call
        tool_call_valid, result, actual_response = self._extract_validation_from_response(response)

        if is_incomplete and tool_call_valid:
            logger.warning("Response was incomplete - treating as invalid")
            tool_call_valid = False

        return tool_call_valid, result, actual_response

    def _extract_validation_from_response(self, response: Any) -> tuple[bool, dict[str, Any], str]:
        """Extract validation results from the API response.

        Args:
            response: OpenAI Responses API response.

        Returns:
            Tuple of (tool_call_valid, result_dict, actual_response_text).
        """
        result: dict[str, Any] = {"keep": [], "adjust": [], "remove": []}
        tool_call_valid = False
        actual_response_parts = []

        for item in response.output:
            if item.type == "function_call" and item.name == "submit_validation":
                raw_args = item.arguments
                try:
                    if isinstance(raw_args, dict):
                        args = raw_args
                    elif isinstance(raw_args, str):
                        args = json.loads(raw_args)
                    else:
                        logger.warning(f"Unexpected arguments type: {type(raw_args)}")
                        tool_call_valid = False
                        break

                    result["keep"] = args.get("keep", [])
                    result["adjust"] = args.get("adjust", [])
                    result["remove"] = args.get("remove", [])

                    if (
                        not isinstance(result["keep"], list)
                        or not isinstance(result["adjust"], list)
                        or not isinstance(result["remove"], list)
                    ):
                        logger.warning("Invalid validation result structure")
                        tool_call_valid = False
                        break

                    tool_call_valid = True
                    break

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse validation JSON: {e}")
                    tool_call_valid = False
                    break
                except (TypeError, AttributeError) as e:
                    logger.warning(f"Error accessing validation arguments: {e}")
                    tool_call_valid = False
                    break

            elif item.type == "message" and hasattr(item, "content"):
                for content_item in item.content:
                    if hasattr(content_item, "text"):
                        actual_response_parts.append(content_item.text)

        actual_response = " ".join(actual_response_parts) if actual_response_parts else "(no text output)"

        return tool_call_valid, result, actual_response
