"""Streamlit dashboard for comparing agent vs WANDS judgments."""

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from goldendemo.config import settings
from goldendemo.data.wands_loader import WANDSLoader
from goldendemo.evaluation import ComparisonResult, compare, list_golden_sets, load_golden_set

# Page config
st.set_page_config(
    page_title="Golden Set Evaluation",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Agent vs WANDS Comparison")


@st.cache_resource
def get_wands_loader() -> WANDSLoader:
    """Load WANDS data (cached)."""
    return WANDSLoader(settings.wands_dir)


@st.cache_data
def get_comparison(filepath: str) -> ComparisonResult:
    """Load and compare a golden set (cached)."""
    loader = get_wands_loader()
    golden_set = load_golden_set(Path(filepath))
    return compare(golden_set, loader)


def score_to_label(score: int) -> str:
    """Convert score to label."""
    return {2: "Exact", 1: "Partial", 0: "Irrelevant"}.get(score, "Unknown")


def render_overlap_chart(result: ComparisonResult) -> None:
    """Render overlap summary as a horizontal bar chart."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=["Products"],
            x=[len(result.only_wands)],
            name=f"Only WANDS ({len(result.only_wands)})",
            orientation="h",
            marker_color="#ff6b6b",
        )
    )

    fig.add_trace(
        go.Bar(
            y=["Products"],
            x=[len(result.both_relevant)],
            name=f"Both ({len(result.both_relevant)})",
            orientation="h",
            marker_color="#51cf66",
        )
    )

    fig.add_trace(
        go.Bar(
            y=["Products"],
            x=[len(result.only_agent)],
            name=f"Only Agent ({len(result.only_agent)})",
            orientation="h",
            marker_color="#339af0",
        )
    )

    fig.update_layout(
        barmode="stack",
        height=150,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Number of Relevant Products",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_score_distribution(result: ComparisonResult) -> None:
    """Render score distribution comparison."""
    # Count scores for each system
    wands_counts = {0: 0, 1: 0, 2: 0}
    agent_counts = {0: 0, 1: 0, 2: 0}

    for score in result.wands_judgments.values():
        wands_counts[score] = wands_counts.get(score, 0) + 1

    for score in result.agent_judgments.values():
        agent_counts[score] = agent_counts.get(score, 0) + 1

    labels = ["Irrelevant (0)", "Partial (1)", "Exact (2)"]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=labels,
            y=[wands_counts[0], wands_counts[1], wands_counts[2]],
            name="WANDS",
            marker_color="#ff6b6b",
        )
    )

    fig.add_trace(
        go.Bar(
            x=labels,
            y=[agent_counts[0], agent_counts[1], agent_counts[2]],
            name="Agent",
            marker_color="#339af0",
        )
    )

    fig.update_layout(
        barmode="group",
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis_title="Count",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_product_table(
    product_ids: set[str],
    result: ComparisonResult,
    loader: WANDSLoader,
    show_wands: bool = True,
    show_agent: bool = True,
) -> None:
    """Render a table of products with their scores."""
    if not product_ids:
        st.info("No products in this category.")
        return

    rows = []
    for pid in sorted(product_ids):
        product = loader.get_product(pid)
        name = product.product_name if product else f"Product {pid}"
        wands_score = result.wands_judgments.get(pid)
        agent_score = result.agent_judgments.get(pid)

        row = {
            "Product ID": pid,
            "Name": name[:60] + "..." if len(name) > 60 else name,
        }
        if show_wands:
            row["WANDS"] = score_to_label(wands_score) if wands_score is not None else "-"
        if show_agent:
            row["Agent"] = score_to_label(agent_score) if agent_score is not None else "-"

        rows.append(row)

    st.dataframe(rows, use_container_width=True, hide_index=True)


def main() -> None:
    """Main dashboard."""
    loader = get_wands_loader()

    # Get available golden sets
    golden_sets_dir = Path(settings.golden_sets_dir)
    golden_set_files = list_golden_sets(golden_sets_dir)

    if not golden_set_files:
        st.warning("No golden sets found. Run the agent first to generate some.")
        st.code(f"Directory checked: {golden_sets_dir}")
        return

    # Query selector
    st.sidebar.header("Select Golden Set")
    selected_file = st.sidebar.selectbox(
        "Golden Set",
        golden_set_files,
        format_func=lambda p: p.stem,
    )

    if not selected_file:
        return

    # Load and compare
    result = get_comparison(str(selected_file))

    # Header with query info
    st.header(f'Query: "{result.query}"')
    if result.query_id:
        st.caption(f"WANDS Query ID: {result.query_id}")
    else:
        st.warning("Could not find matching WANDS query - comparison limited to agent data only")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("WANDS Relevant", len(result.wands_relevant))
    with col2:
        st.metric("Agent Relevant", len(result.agent_relevant))
    with col3:
        st.metric("Both Agree", len(result.both_relevant))
    with col4:
        st.metric("Coverage", f"{result.agent_coverage:.1%}")

    # Overlap visualization
    st.subheader("Overlap Summary")
    render_overlap_chart(result)

    # Score distribution
    st.subheader("Score Distribution")
    render_score_distribution(result)

    # Product lists in tabs
    st.subheader("Product Details")
    tab1, tab2, tab3 = st.tabs(
        [
            f"Both Relevant ({len(result.both_relevant)})",
            f"Only WANDS ({len(result.only_wands)})",
            f"Only Agent ({len(result.only_agent)})",
        ]
    )

    with tab1:
        st.caption("Products both systems found relevant")
        render_product_table(result.both_relevant, result, loader)

    with tab2:
        st.caption("Products WANDS found relevant but agent missed or marked irrelevant")
        render_product_table(result.only_wands, result, loader, show_agent=True)

    with tab3:
        st.caption("Products agent found relevant but not in WANDS (or WANDS marked irrelevant)")
        render_product_table(result.only_agent, result, loader, show_wands=True)


if __name__ == "__main__":
    main()
