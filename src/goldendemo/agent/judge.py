"""Judgment subagent for isolated product evaluation.

This subagent evaluates products in isolation to prevent context window explosion.
Each call gets fresh context with just the query and products to judge.
"""

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from goldendemo.agent.state import AgentState

from openai import OpenAI

from goldendemo.config import settings

logger = logging.getLogger(__name__)


# Judgment prompt for subagent
JUDGMENT_PROMPT = """You are a product relevance judge for e-commerce search using WANDS methodology.

## Your Task
Query: "{query}"

Judge each product below as **Exact (2)** or **Partial (1)**.

## CRITICAL: You MUST respond with a tool call
- **Your ONLY valid response is calling the submit_judgments tool**
- **DO NOT return text** - only tool calls are accepted

## WANDS Scoring Methodology

Every query has a **target entity** (the main product type) and **modifiers** (descriptive attributes).

### Exact (2) - Fully matches query
Product has BOTH the target entity AND all modifiers.

**Examples:**
- Query: "modern sofa" → Modern sofa (has sofa + modern)
- Query: "driftwood mirror" → Driftwood-framed mirror (has mirror + driftwood)
- Query: "blue velvet chair" → Blue velvet chair (has chair + blue + velvet)

### Partial (1) - Matches target entity, missing/wrong modifiers
Product has the TARGET ENTITY but does NOT satisfy all modifiers.

**BE GENEROUS with Partial!** If it's the right product type but wrong style/color/material, it's Partial.

**Examples:**
- Query: "modern sofa" → Traditional sofa (has sofa ✓, missing modern ✗)
- Query: "modern sofa" → Vintage sofa (has sofa ✓, wrong style ✗)
- Query: "modern sofa" → Contemporary sofa (has sofa ✓, similar but not exact ✗)
- Query: "blue velvet chair" → Red velvet chair (has chair ✓, wrong color ✗)
- Query: "blue velvet chair" → Blue leather chair (has blue chair ✓, wrong material ✗)
- Query: "driftwood mirror" → Oak mirror (has mirror ✓, wrong material ✗)
- Query: "driftwood mirror" → Wall mirror (has mirror ✓, missing material ✗)
- Query: "driftwood mirror" → Gold-framed mirror (has mirror ✓, wrong material ✗)

### Skip - Missing target entity
Product does NOT have the target entity.

**Examples:**
- Query: "modern sofa" → Modern table (no sofa, skip)
- Query: "driftwood mirror" → Driftwood table (no mirror, skip)
- Query: "blue velvet chair" → Blue velvet curtains (no chair, skip)

## Decision Framework

For EACH product:

1. **Does it have the TARGET ENTITY from the query?**
   - NO → Skip (do not include in judgments)
   - YES → Continue

2. **Does it match ALL modifiers exactly?**
   - YES → Exact (2)
   - NO → Partial (1)

**Critical**: If a customer searching for "{query}" would recognize this as the same TYPE of product (even if wrong color/style/material), mark it Partial!

---

## Products to Judge

{products_list}

---

## CRITICAL: Tool-Only Operation

You MUST respond with a tool call. DO NOT return text.

**Call submit_judgments with judgments ONLY for products that are Exact or Partial. Skip unrelated products.**

⚠️ IMPORTANT: Use the EXACT PRODUCT_ID shown for each product. Do NOT make up IDs, use sequential numbers, or use the product number (#1, #2, etc.). Copy the PRODUCT_ID field exactly as shown.

If you return text instead of a tool call, the system will fail. You have ONE job: call the submit_judgments tool with your relevance judgments for relevant products only."""


def _format_products_for_judgment(products: list[dict]) -> str:
    """Format products for the judgment prompt.

    CRITICAL: We must use product_id from the data, not the numbering.
    """
    lines = []
    for i, p in enumerate(products, 1):
        # Make product_id VERY clear - it's not the number, it's the ID field
        product_id = p["product_id"]
        lines.append(f"--- Product #{i} ---")
        lines.append(f"PRODUCT_ID: {product_id}  ← USE THIS EXACT ID IN YOUR JUDGMENT")
        lines.append(f"Name: {p['product_name']}")
        lines.append(f"Category: {p.get('category', p.get('product_class', 'Unknown'))}")
        if p.get("description"):
            desc = p["description"]
            # Truncate long descriptions
            if len(desc) > 200:
                desc = desc[:200] + "..."
            lines.append(f"Description: {desc}")
        if p.get("attributes"):
            attrs = p["attributes"]
            if attrs:
                attrs_str = ", ".join(f"{k}: {', '.join(v) if isinstance(v, list) else v}" for k, v in attrs.items())
                lines.append(f"Attributes: {attrs_str}")
        lines.append("")
    return "\n".join(lines)


class JudgmentSubagent:
    """Subagent for judging product relevance in isolation.

    This subagent receives a query and a list of products, and returns
    relevance judgments. Each call has fresh context to prevent context
    window explosion.
    """

    def __init__(
        self,
        openai_client: OpenAI | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ):
        """Initialize the judgment subagent.

        Args:
            openai_client: OpenAI client. Creates one if not provided.
            model: Model to use (defaults to JUDGE_MODEL from .env).
            reasoning_effort: Reasoning effort level (defaults to JUDGE_REASONING_EFFORT from .env).
        """
        self.openai_client = openai_client or OpenAI(api_key=settings.openai_api_key)
        self.model = model if model is not None else settings.judge_model
        self.reasoning_effort = reasoning_effort if reasoning_effort is not None else settings.judge_reasoning_effort

        # Define the submit_judgments tool for structured output
        # strict=True ensures reliable schema adherence per OpenAI docs
        self.tools = [
            {
                "type": "function",
                "name": "submit_judgments",
                "description": "Submit relevance judgments for products that are Exact or Partial matches. Return empty array if no products are relevant.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "judgments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "product_id": {
                                        "type": "string",
                                        "description": "Product ID being judged",
                                    },
                                    "relevance": {
                                        "type": "integer",
                                        "enum": [1, 2],
                                        "description": "Relevance score: 1=Partial, 2=Exact",
                                    },
                                    "reasoning": {
                                        "type": "string",
                                        "description": "Brief explanation for this relevance judgment",
                                    },
                                },
                                "required": ["product_id", "relevance", "reasoning"],
                                "additionalProperties": False,
                            },
                            "description": "List of relevance judgments. Empty array is valid if no products are Exact or Partial matches.",
                        },
                    },
                    "required": ["judgments"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]

    def judge_products(self, query: str, products: list[dict], state: "AgentState | None" = None) -> list[dict]:
        """Judge relevance of products for a query with retry guardrail.

        Args:
            query: Search query.
            products: List of product dicts to judge.
            state: Optional agent state for token tracking.

        Returns:
            List of judgment dicts: [{"product_id": str, "relevance": int, "reasoning": str}]

        Raises:
            RuntimeError: If subagent fails to call tool after retries.
        """
        if not products:
            logger.warning("judge_products called with empty product list")
            return []

        logger.info(f"Subagent judging {len(products)} products for query '{query}'")

        # Retry if model doesn't call tool or payload is invalid
        max_attempts = settings.judge_max_retries
        for attempt in range(1, max_attempts + 1):
            try:
                tool_call_valid, judgments, response_text = self._call_and_extract_judgments(
                    query, products, attempt, state
                )

                if tool_call_valid:
                    # Tool was called successfully with valid payload - empty judgments is valid (no relevant products)
                    if judgments:
                        logger.info(
                            f"Subagent successfully judged {len(judgments)}/{len(products)} products on attempt {attempt}"
                        )
                    else:
                        logger.info(
                            f"Subagent returned 0 judgments (all {len(products)} products deemed irrelevant) on attempt {attempt}"
                        )
                    return judgments
                else:
                    # Tool was NOT called or payload was invalid (parse failure, truncation, etc.)
                    logger.warning(f"Attempt {attempt}/{max_attempts}: Subagent tool call invalid or missing")
                    logger.warning(f"Subagent response: {response_text[:200]}")
                    if attempt < max_attempts:
                        logger.info("Retrying with stronger prompt...")
                        continue
                    else:
                        raise RuntimeError(
                            f"Subagent failed to call submit_judgments tool with valid payload after {max_attempts} attempts. "
                            f"Possible causes: tool not called, invalid JSON, truncated response, or schema mismatch."
                        )

            except Exception as e:
                if attempt < max_attempts:
                    logger.warning(f"Attempt {attempt} failed: {e}. Retrying...")
                    continue
                else:
                    logger.error(f"Judgment subagent failed after {max_attempts} attempts: {e}", exc_info=True)
                    raise

        # Should never reach here, but just in case
        raise RuntimeError(f"Subagent failed to produce judgments after {max_attempts} attempts")

    def _call_and_extract_judgments(
        self, query: str, products: list[dict], attempt: int, state: "AgentState | None" = None
    ) -> tuple[bool, list[dict], str]:
        """Make API call and extract judgments.

        Args:
            query: Search query.
            products: Products to judge.
            attempt: Attempt number (for stronger prompting on retries).
            state: Optional agent state for token tracking.

        Returns:
            Tuple of (found_tool_call, judgment dicts list, actual response text).
        """
        # Format products for prompt
        products_list = _format_products_for_judgment(products)

        # Build prompt (same for all attempts - tool calling is reliable now)
        prompt = JUDGMENT_PROMPT.format(
            query=query,
            products_list=products_list,
            product_count=len(products),
        )

        # Call OpenAI Responses API
        logger.debug(f"Attempt {attempt}: Calling OpenAI with model={self.model}, effort={self.reasoning_effort}")
        logger.debug(f"Judging {len(products)} products: {[p['product_id'] for p in products]}")

        # Subagent should ONLY call tools, no reasoning summaries allowed
        # Force the model to call the specific submit_judgments function
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
            tool_choice={"type": "function", "name": "submit_judgments"},  # Force this specific tool
            reasoning={"effort": self.reasoning_effort},
            max_output_tokens=settings.judge_max_output_tokens,
        )

        # Check response status for incomplete/truncated responses
        response_status = getattr(response, "status", "unknown")
        is_incomplete = response_status == "incomplete"

        if is_incomplete:
            logger.warning("Response status is 'incomplete' - response was truncated")
        elif response_status not in ("completed", "unknown"):
            logger.warning(f"Unexpected response status: {response_status}")

        # Track token usage if state provided
        if state and hasattr(response, "usage") and response.usage:
            usage_dict = response.usage.model_dump() if hasattr(response.usage, "model_dump") else response.usage
            state.token_usage.add_usage(usage_dict)
            logger.debug(
                f"Subagent token usage: {usage_dict.get('total_tokens', 0)} "
                f"(input: {usage_dict.get('input_tokens', 0)}, "
                f"output: {usage_dict.get('output_tokens', 0)})"
            )

        # Extract judgments from tool call
        tool_call_valid, judgments, actual_response = self._extract_judgments_from_response(response)

        # If response was incomplete, treat it as invalid (trigger retry)
        if is_incomplete and tool_call_valid:
            logger.warning("Response was incomplete - treating as invalid even though tool was called")
            tool_call_valid = False

        return tool_call_valid, judgments, actual_response

    def _extract_judgments_from_response(self, response: Any) -> tuple[bool, list[dict], str]:
        """Extract judgments from the API response.

        Args:
            response: OpenAI Responses API response.

        Returns:
            Tuple of (tool_call_valid, judgments list, actual_response_text).
            tool_call_valid=True only if tool was called AND payload was parseable.
            Empty judgments with tool_call_valid=True means "no relevant products" (success).
            Empty judgments with tool_call_valid=False means parse failure or no tool call (retry).
        """
        judgments = []
        tool_call_valid = False
        actual_response_parts = []

        for item in response.output:
            if item.type == "function_call" and item.name == "submit_judgments":
                # Tool was called - now check if arguments are valid
                raw_args = item.arguments  # Get arguments before try block
                try:
                    # Handle arguments being dict (already parsed) or string (needs parsing)
                    if isinstance(raw_args, dict):
                        args = raw_args
                    elif isinstance(raw_args, str):
                        args = json.loads(raw_args)
                    else:
                        logger.warning(f"Tool called but arguments have unexpected type: {type(raw_args)}")
                        tool_call_valid = False
                        break

                    # Extract judgments
                    judgments = args.get("judgments", [])
                    if not isinstance(judgments, list):
                        logger.warning(f"Tool called but 'judgments' is not a list: {type(judgments)}")
                        tool_call_valid = False
                        break

                    # Success - valid tool call with valid payload
                    tool_call_valid = True
                    break

                except json.JSONDecodeError as e:
                    # Tool was called but arguments are malformed JSON
                    logger.warning(f"Tool called but failed to parse JSON arguments: {e}")
                    logger.debug(f"Invalid JSON arguments (first 500 chars): {str(raw_args)[:500]}")
                    tool_call_valid = False
                    break
                except (TypeError, AttributeError) as e:
                    # Unexpected error accessing arguments
                    logger.warning(f"Tool called but error accessing arguments: {e}")
                    tool_call_valid = False
                    break
            elif item.type == "message" and hasattr(item, "content"):
                # Capture text responses
                for content_item in item.content:
                    if hasattr(content_item, "text"):
                        actual_response_parts.append(content_item.text)
            elif item.type == "reasoning" and hasattr(item, "summary"):
                # Capture reasoning summaries
                for summary_item in item.summary:
                    if hasattr(summary_item, "text"):
                        actual_response_parts.append(f"[Reasoning: {summary_item.text}]")

        actual_response = " ".join(actual_response_parts) if actual_response_parts else "(no text output)"

        # If tool call invalid, log what the model actually returned
        if not tool_call_valid:
            # Log detailed response structure only at DEBUG level to avoid verbose output
            logger.debug("=" * 60)
            logger.debug("JUDGMENT SUBAGENT TOOL CALL INVALID OR MISSING")
            logger.debug("=" * 60)
            logger.debug(f"Response status: {getattr(response, 'status', 'unknown')}")
            logger.debug(f"Output items count: {len(response.output) if response.output else 0}")

            for idx, item in enumerate(response.output):
                logger.debug(f"  Item {idx}: type={item.type}")
                if item.type == "reasoning":
                    if hasattr(item, "summary") and item.summary:
                        summary_texts = [s.text[:200] for s in item.summary if hasattr(s, "text")]
                        logger.debug(f"    Reasoning summary: {' '.join(summary_texts) if summary_texts else 'none'}")
                    else:
                        logger.debug("    Reasoning: (encrypted or no summary)")
                elif item.type == "message":
                    if hasattr(item, "content") and item.content:
                        # Extract text from content items
                        content_texts = []
                        for c in item.content:
                            if hasattr(c, "text"):
                                content_texts.append(c.text[:200])
                        logger.debug(f"    Message content: {' '.join(content_texts) if content_texts else 'empty'}")
                    else:
                        logger.debug("    Message: no content")

            logger.debug(f"Captured text: {actual_response[:500]}")
            logger.debug("=" * 60)

        return tool_call_valid, judgments, actual_response

    def _log_failed_response(self, response: Any) -> None:
        """Log what the model returned when it didn't call the tool.

        Args:
            response: OpenAI Responses API response.
        """
        logger.warning("=" * 60)
        logger.warning("SUBAGENT FAILED TO CALL TOOL - Response content:")
        logger.warning("=" * 60)

        for item in response.output:
            if item.type == "reasoning":
                # Show reasoning summary if available
                if hasattr(item, "summary") and item.summary:
                    summaries = [s.text for s in item.summary if hasattr(s, "text")]
                    if summaries:
                        logger.warning(f"[REASONING] {' '.join(summaries)[:500]}")
            elif item.type == "message":
                # Show message content
                if item.content:
                    text = item.content[0].text if item.content else ""
                    if text:
                        logger.warning(f"[MESSAGE] {text[:500]}")
            elif item.type == "function_call":
                # Show which function it called (if any)
                logger.warning(f"[TOOL CALL] {item.name} (wrong tool!)")

        logger.warning("=" * 60)
