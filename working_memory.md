# Working Memory: Agent Development Journey

This document tracks the problems encountered and solutions implemented while building the golden set agent.

---

## Problem 1: Judgment Overwrites (SOLVED)

**Issue**: Agent replaced all previous judgments on each `submit_judgments` call instead of accumulating them.

**Solution**: Changed `state.judgments = judgments` to additive mode where new judgments are appended (deduplicating by product_id).

---

## Problem 2: AttributeError in Tool Results (SOLVED)

**Issue**: `'list' object has no attribute 'get'` when accessing `result.data.get("result_count")`.

**Root Cause**: `ToolResult.data` is a list for browse/search results, not a dict. Extra metadata like `result_count` is stored in `ToolResult.metadata`.

**Solution**: Changed `result.data.get("result_count")` to `result.metadata.get("result_count")`.

---

## Problem 3: Infinite Loop - Agent Never Submits Judgments (SOLVED)

**Issue**: Agent kept browsing the same offset getting 0 results, never calling `complete_step()` or submitting judgments.

**Root Cause**: Context was resetting EVERY iteration in execution phase. Agent couldn't see browse results to judge them because the results were cleared before it could process them.

**Solution**: Only reset context at step boundaries (when `complete_step()` is called), not every iteration. Accumulate context within a step.

---

## Problem 4: Invalid Category Names (SOLVED)

**Issue**: Agent tried to browse invalid categories like "Jewelry Armoires" and "Makeup Vanities|Desks". These came from `category_hierarchy` paths, not actual `product_class` values.

**Attempted Fix**: Add fuzzy matching to map invalid names to valid categories.

**User Feedback**: "Stop - are you sure this is the right way? I don't want to add yet more business logic to this demo app."

**Final Solution**: Remove `category_hierarchy` from search results entirely. Rename `category` to `product_class` in output to match `list_categories`. Agent now uses consistent naming.

---

## Problem 5: Context Window Explosion (CURRENT)

**Issue**: Even with step-boundary resets, context still exceeds limits. A category with 416 products requires 5 pages of 100 products each. Each page adds ~20KB of product data to context. Within a single step, context grows to 100KB+.

**Error**: `context_length_exceeded` after processing multiple pages within one category step.

**Root Cause**: The main agent accumulates product data in its conversation history across pages within the same step. Even though we reset between steps, a single large step blows out context.

**User Direction**: "We explicitly said to not collect product data across runs. The page that the agent is on is the only page that matters. We should divorce the context there, giving it only the information it needs to make the judgment for that page of results. In Claude Code, it's done by delegating tasks to subagents."

---

## Solution: Subagent Delegation Pattern

### Architecture

```
MAIN AGENT (Orchestrator)
- Handles discovery: list_categories, search_products
- Creates plan via submit_plan
- Browses categories page by page
- Delegates judgment to subagent
- Never sees product data in context
                    |
                    v (for each page)
JUDGMENT SUBAGENT (Isolated)
- Receives: query + 100 products
- Returns: [{product_id, score, reasoning}]
- Fresh context each call (~15KB)
```

### Key Change

Judgment moves INSIDE the `browse_category` tool:
1. Tool fetches products from Weaviate
2. Tool calls subagent to judge (isolated API call)
3. Tool returns summary: "Browsed 100 products, added 45 judgments"
4. Main agent NEVER sees product data

### Files to Modify

| File | Change |
|------|--------|
| `src/goldendemo/agent/judge.py` | NEW: JudgmentSubagent class |
| `src/goldendemo/agent/tools/browse.py` | Integrate subagent into browse_category |
| `src/goldendemo/agent/tools/submit.py` | DELETE (judgments happen in browse) |
| `src/goldendemo/agent/agent.py` | Initialize subagent, simplify execution |
| `src/goldendemo/agent/prompts.py` | Simplify execution prompt |
| `src/goldendemo/agent/state.py` | Add `add_judgments_from_dicts()` |
| `src/goldendemo/config.py` | Add `JUDGE_MODEL` setting |

### Expected Result

- Main agent context: ~5KB per iteration (plan + tool result summary)
- Subagent context: ~15KB per page (products + prompt)
- No accumulation between pages or steps
- Can process categories of any size

---

## Lessons Learned

1. **Don't accumulate what you don't need**: Product data is only needed for judgment, not for orchestration
2. **Isolate expensive operations**: Use subagents for tasks that require large context
3. **Keep the main agent lean**: It should orchestrate, not process
4. **Avoid adding business logic**: Simple data format changes beat complex fuzzy matching
