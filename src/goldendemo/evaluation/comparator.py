"""Compare agent golden sets against WANDS ground truth."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from goldendemo.data.models import GoldenSetConfig
from goldendemo.data.wands_loader import WANDSLoader


@dataclass
class ComparisonResult:
    """Result of comparing agent judgments to WANDS ground truth."""

    query: str
    query_id: str | None

    # What each found relevant (score >= 1)
    wands_relevant: set[str] = field(default_factory=set)
    agent_relevant: set[str] = field(default_factory=set)

    # Intersection analysis
    both_relevant: set[str] = field(default_factory=set)
    only_wands: set[str] = field(default_factory=set)
    only_agent: set[str] = field(default_factory=set)

    # Score-level agreement (for products both found relevant)
    exact_agreement: set[str] = field(default_factory=set)  # Both said Exact (2)
    partial_agreement: set[str] = field(default_factory=set)  # Both said Partial (1)
    agent_upgraded: set[str] = field(default_factory=set)  # Agent=Exact, WANDS=Partial
    agent_downgraded: set[str] = field(default_factory=set)  # Agent=Partial, WANDS=Exact

    # Full judgment maps for detail view
    wands_judgments: dict[str, int] = field(default_factory=dict)
    agent_judgments: dict[str, int] = field(default_factory=dict)

    @property
    def agent_coverage(self) -> float:
        """What % of WANDS relevant products did agent also find relevant."""
        if not self.wands_relevant:
            return 0.0
        return len(self.both_relevant) / len(self.wands_relevant)

    @property
    def agent_precision(self) -> float:
        """What % of agent's relevant products are also WANDS relevant."""
        if not self.agent_relevant:
            return 0.0
        return len(self.both_relevant) / len(self.agent_relevant)

    @property
    def score_agreement_rate(self) -> float:
        """What % of overlapping products have exact score match."""
        if not self.both_relevant:
            return 0.0
        exact_matches = len(self.exact_agreement) + len(self.partial_agreement)
        return exact_matches / len(self.both_relevant)


def load_golden_set(filepath: Path) -> GoldenSetConfig:
    """Load a golden set from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return GoldenSetConfig.model_validate(data)


def find_wands_query_id(query_text: str, loader: WANDSLoader) -> str | None:
    """Find WANDS query_id by matching query text."""
    query_text_lower = query_text.lower().strip()
    for q in loader.get_all_queries():
        if q.query.lower().strip() == query_text_lower:
            return q.query_id
    return None


def compare(golden_set: GoldenSetConfig, loader: WANDSLoader) -> ComparisonResult:
    """Compare agent golden set against WANDS ground truth.

    Args:
        golden_set: Agent-generated golden set.
        loader: WANDS data loader.

    Returns:
        ComparisonResult with overlap analysis.
    """
    # Find WANDS query_id from golden set query text
    query_id: str | None = golden_set.query_id
    if query_id and not query_id.isdigit():
        # query_id might be the query text, not the numeric ID
        query_id = find_wands_query_id(golden_set.query, loader)

    if not query_id:
        query_id = find_wands_query_id(golden_set.query, loader)

    # Get WANDS ground truth
    wands_scores = loader.get_ground_truth_scores(query_id) if query_id else {}

    # Get agent judgments
    agent_scores = golden_set.get_relevance_map()

    # Find relevant products (score >= 1 means Exact or Partial)
    wands_relevant = {pid for pid, score in wands_scores.items() if score >= 1}
    agent_relevant = {pid for pid, score in agent_scores.items() if score >= 1}

    # Compute intersections
    both_relevant = wands_relevant & agent_relevant
    only_wands = wands_relevant - agent_relevant
    only_agent = agent_relevant - wands_relevant

    # Compute score-level agreement for overlapping products
    exact_agreement: set[str] = set()
    partial_agreement: set[str] = set()
    agent_upgraded: set[str] = set()
    agent_downgraded: set[str] = set()

    for pid in both_relevant:
        wands_score = wands_scores.get(pid, 0)
        agent_score = agent_scores.get(pid, 0)

        if wands_score == 2 and agent_score == 2:
            exact_agreement.add(pid)
        elif wands_score == 1 and agent_score == 1:
            partial_agreement.add(pid)
        elif wands_score == 1 and agent_score == 2:
            agent_upgraded.add(pid)  # Agent more generous
        elif wands_score == 2 and agent_score == 1:
            agent_downgraded.add(pid)  # Agent less generous

    return ComparisonResult(
        query=golden_set.query,
        query_id=query_id,
        wands_relevant=wands_relevant,
        agent_relevant=agent_relevant,
        both_relevant=both_relevant,
        only_wands=only_wands,
        only_agent=only_agent,
        exact_agreement=exact_agreement,
        partial_agreement=partial_agreement,
        agent_upgraded=agent_upgraded,
        agent_downgraded=agent_downgraded,
        wands_judgments=wands_scores,
        agent_judgments=agent_scores,
    )


def list_golden_sets(directory: Path) -> list[Path]:
    """List all golden set JSON files in a directory."""
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
