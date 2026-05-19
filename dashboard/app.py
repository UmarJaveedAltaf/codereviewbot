"""CodeReviewBot — Streamlit Dashboard.

Four pages navigated via the left sidebar:

    1. Overview          — KPI cards, issues-per-day chart, category pie chart.
    2. Repository Breakdown — Per-repo table with acceptance rates.
    3. Review History    — Searchable, filterable table of all stored reviews.
    4. Convention Effectiveness — Category acceptance rates and rule listing.

Data is loaded from the two ChromaDB collections (past_reviews and
team_conventions) with a 5-minute cache TTL so the dashboard stays
responsive without hammering the database.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys

# Make the project root importable when running via `streamlit run`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import settings

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CodeReviewBot Dashboard",
    page_icon="🤖",
    layout="wide",
)

# ── Colour palette (matches severity icons) ───────────────────────────────────

_SEVERITY_COLOURS = {
    "CRITICAL":   "#FF4444",
    "critical":   "#FF4444",
    "WARNING":    "#FFB800",
    "high":       "#FF8C00",
    "medium":     "#FFB800",
    "low":        "#4CAF50",
    "SUGGESTION": "#4CAF50",
    "info":       "#2196F3",
}

_CATEGORY_COLOURS = [
    "#EF5350", "#AB47BC", "#42A5F5",
    "#26A69A", "#FFA726", "#66BB6A",
]


# ── Data loading (cached, 5-minute TTL) ───────────────────────────────────────


@st.cache_data(ttl=300)
def load_reviews() -> pd.DataFrame:
    """Load all records from the ``past_reviews`` ChromaDB collection.

    Returns an empty ``DataFrame`` when the collection is empty.
    Columns: repo, file, severity, category, accepted, timestamp, comment, snippet.
    """
    try:
        client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        col    = client.get_or_create_collection("past_reviews")

        if col.count() == 0:
            return pd.DataFrame(columns=[
                "repo", "file", "severity", "category",
                "accepted", "timestamp", "comment", "snippet", "date",
            ])

        results = col.get(include=["documents", "metadatas"])
        rows = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            rows.append({
                "repo":      meta.get("repo", ""),
                "file":      meta.get("file", ""),
                "severity":  meta.get("severity", "WARNING"),
                "category":  meta.get("category", "general"),
                "accepted":  int(meta.get("accepted", -1)),
                "timestamp": meta.get("timestamp", ""),
                "comment":   meta.get("comment", "")[:200],
                "snippet":   doc[:100],
            })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
        return df

    except Exception as exc:
        st.error(f"Failed to load reviews from ChromaDB: {exc}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_conventions() -> pd.DataFrame:
    """Load all records from the ``team_conventions`` ChromaDB collection.

    Returns an empty ``DataFrame`` when the collection is empty.
    Columns: rule, category, language, repo, added_by.
    """
    try:
        client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        col    = client.get_or_create_collection("team_conventions")

        if col.count() == 0:
            return pd.DataFrame(columns=[
                "rule", "category", "language", "repo", "added_by",
            ])

        results = col.get(include=["documents", "metadatas"])
        rows = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            rows.append({
                "rule":      doc,
                "category":  meta.get("category", "general"),
                "language":  meta.get("language", "") or "all",
                "repo":      meta.get("repo", "")    or "global",
                "added_by":  meta.get("added_by", "system"),
            })

        return pd.DataFrame(rows)

    except Exception as exc:
        st.error(f"Failed to load conventions from ChromaDB: {exc}")
        return pd.DataFrame()


# ── Shared KPI helpers ────────────────────────────────────────────────────────


def _acceptance_rate(df: pd.DataFrame, repo: str | None = None) -> float:
    """Fraction of rated reviews that were accepted (excludes unrated)."""
    if df.empty or "accepted" not in df.columns:
        return 0.0
    subset = df if repo is None else df[df["repo"] == repo]
    rated  = subset[subset["accepted"] != -1]
    if rated.empty:
        return 0.0
    return (rated["accepted"] == 1).sum() / len(rated)


def _severity_counts(df: pd.DataFrame) -> dict[str, int]:
    """Count reviews by normalised severity label."""
    mapping = {
        "CRITICAL":   "critical",
        "WARNING":    "high",
        "SUGGESTION": "low",
    }
    normalised = df["severity"].map(
        lambda s: mapping.get(str(s).upper(), str(s).lower())
    )
    return normalised.value_counts().to_dict()


# ── Page 1 — Overview ─────────────────────────────────────────────────────────


def page_overview(df: pd.DataFrame) -> None:
    st.header("📊 Overview")

    # ── KPI cards ────────────────────────────────────────────────────────
    total_reviews  = len(df)
    total_repos    = df["repo"].nunique() if not df.empty else 0
    accept_rate    = _acceptance_rate(df)
    sev_counts     = _severity_counts(df)
    critical_count = sev_counts.get("critical", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", total_reviews)
    c2.metric("Repositories", total_repos)
    c3.metric("Acceptance Rate", f"{accept_rate:.0%}")
    c4.metric("🚨 Critical Issues", critical_count,
              delta=None,
              delta_color="inverse" if critical_count > 0 else "off")

    if df.empty:
        st.info(
            "No reviews stored yet. Trigger a PR review to populate the dashboard."
        )
        return

    st.divider()

    col_chart, col_pie = st.columns([3, 2])

    # ── Issues per day (line chart) ───────────────────────────────────────
    with col_chart:
        st.subheader("Issues per Day")
        daily = (
            df.dropna(subset=["date"])
            .groupby("date")
            .size()
            .reset_index(name="issues")
        )
        if not daily.empty:
            daily["date"] = pd.to_datetime(daily["date"])
            fig = px.line(
                daily, x="date", y="issues",
                labels={"date": "Date", "issues": "Issues Found"},
                markers=True,
                color_discrete_sequence=["#42A5F5"],
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#333"),
                yaxis=dict(gridcolor="#333"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No date information available for the chart.")

    # ── Issues by category (pie chart) ───────────────────────────────────
    with col_pie:
        st.subheader("Issues by Category")
        cat_counts = df["category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        if not cat_counts.empty:
            fig = px.pie(
                cat_counts, names="category", values="count",
                color_discrete_sequence=_CATEGORY_COLOURS,
                hole=0.35,
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.15),
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

    # ── Severity breakdown bar ────────────────────────────────────────────
    st.subheader("Severity Breakdown")
    sev_df = pd.DataFrame(
        list(sev_counts.items()), columns=["severity", "count"]
    ).sort_values("count", ascending=False)

    if not sev_df.empty:
        sev_df["colour"] = sev_df["severity"].map(
            lambda s: _SEVERITY_COLOURS.get(s, "#9E9E9E")
        )
        fig = px.bar(
            sev_df, x="severity", y="count",
            color="severity",
            color_discrete_map=_SEVERITY_COLOURS,
            labels={"severity": "Severity", "count": "Count"},
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Page 2 — Repository Breakdown ─────────────────────────────────────────────


def page_repo_breakdown(df: pd.DataFrame) -> None:
    st.header("🗂️ Repository Breakdown")

    if df.empty:
        st.info("No review data available yet.")
        return

    repos = df["repo"].unique()
    rows  = []
    for repo in sorted(repos):
        rdf          = df[df["repo"] == repo]
        rated        = rdf[rdf["accepted"] != -1]
        accepted_cnt = int((rated["accepted"] == 1).sum())
        rejected_cnt = int((rated["accepted"] == 0).sum())
        sev          = _severity_counts(rdf)

        rows.append({
            "Repository":      repo,
            "Total Reviews":   len(rdf),
            "🚨 Critical":     sev.get("critical", 0),
            "❗ High":         sev.get("high", 0),
            "⚠️ Medium":       sev.get("medium", 0),
            "💡 Low":          sev.get("low", 0),
            "✅ Accepted":     accepted_cnt,
            "❌ Rejected":     rejected_cnt,
            "Acceptance Rate": f"{_acceptance_rate(df, repo):.0%}",
        })

    repo_df = pd.DataFrame(rows)
    st.dataframe(repo_df, use_container_width=True, hide_index=True)

    # ── Per-repo acceptance rate bar ─────────────────────────────────────
    st.subheader("Acceptance Rate by Repository")
    rate_data = pd.DataFrame({
        "repo": [r["Repository"] for r in rows],
        "rate": [_acceptance_rate(df, r["Repository"]) for r in rows],
    }).sort_values("rate", ascending=True)

    fig = px.bar(
        rate_data, x="rate", y="repo", orientation="h",
        labels={"rate": "Acceptance Rate", "repo": "Repository"},
        color="rate",
        color_continuous_scale=["#FF4444", "#FFB800", "#36A64F"],
        range_color=[0, 1],
        text=rate_data["rate"].map("{:.0%}".format),
    )
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


# ── Page 3 — Review History ───────────────────────────────────────────────────


def page_review_history(df: pd.DataFrame) -> None:
    st.header("📜 Review History")

    if df.empty:
        st.info("No reviews stored yet.")
        return

    # ── Filter controls ───────────────────────────────────────────────────
    with st.expander("🔍 Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)

        all_repos = ["All"] + sorted(df["repo"].dropna().unique().tolist())
        repo_filter = fc1.selectbox("Repository", all_repos, key="hist_repo")

        all_sevs = ["All"] + sorted(df["severity"].dropna().unique().tolist())
        sev_filter = fc2.selectbox("Severity", all_sevs, key="hist_sev")

        all_cats = ["All"] + sorted(df["category"].dropna().unique().tolist())
        cat_filter = fc3.selectbox("Category", all_cats, key="hist_cat")

        search = st.text_input(
            "🔎 Search comments",
            placeholder="Type to filter by comment text…",
            key="hist_search",
        )

    # ── Apply filters ─────────────────────────────────────────────────────
    filtered = df.copy()
    if repo_filter != "All":
        filtered = filtered[filtered["repo"] == repo_filter]
    if sev_filter != "All":
        filtered = filtered[filtered["severity"] == sev_filter]
    if cat_filter != "All":
        filtered = filtered[filtered["category"] == cat_filter]
    if search.strip():
        mask = filtered["comment"].str.contains(
            search.strip(), case=False, na=False
        )
        filtered = filtered[mask]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} reviews")

    # ── Acceptance status label ───────────────────────────────────────────
    _ACCEPTED_LABEL = {1: "✅ Accepted", 0: "❌ Rejected", -1: "— Unrated"}
    display = filtered[[
        "repo", "file", "severity", "category", "timestamp", "comment"
    ]].copy()
    display["status"] = filtered["accepted"].map(
        lambda v: _ACCEPTED_LABEL.get(int(v), "— Unrated")
    )
    display = display.rename(columns={
        "repo": "Repository", "file": "File",
        "severity": "Severity", "category": "Category",
        "timestamp": "Timestamp", "comment": "Comment",
        "status": "Status",
    })

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Comment": st.column_config.TextColumn(max_chars=120),
            "Timestamp": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        },
    )


# ── Page 4 — Convention Effectiveness ─────────────────────────────────────────


def page_convention_effectiveness(
    reviews_df: pd.DataFrame,
    conventions_df: pd.DataFrame,
) -> None:
    st.header("📏 Convention Effectiveness")

    if conventions_df.empty:
        st.info(
            "No conventions loaded yet. Run `python scripts/seed_conventions.py` "
            "to seed the team_conventions collection."
        )
        return

    st.metric("Total Conventions", len(conventions_df))

    col_cat, col_lang = st.columns(2)

    # ── Conventions by category ───────────────────────────────────────────
    with col_cat:
        st.subheader("By Category")
        cat_counts = conventions_df["category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        fig = px.bar(
            cat_counts, x="count", y="category", orientation="h",
            color="count",
            color_continuous_scale="Blues",
            labels={"count": "Rules", "category": "Category"},
        )
        fig.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Conventions by language ───────────────────────────────────────────
    with col_lang:
        st.subheader("By Language")
        lang_counts = conventions_df["language"].value_counts().reset_index()
        lang_counts.columns = ["language", "count"]
        fig = px.pie(
            lang_counts, names="language", values="count",
            color_discrete_sequence=_CATEGORY_COLOURS,
            hole=0.35,
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Acceptance rate per review category ───────────────────────────────
    if not reviews_df.empty:
        st.subheader("Review Acceptance Rate by Category")
        cats = reviews_df["category"].unique()
        cat_rows = []
        for cat in sorted(cats):
            cat_df   = reviews_df[reviews_df["category"] == cat]
            rated    = cat_df[cat_df["accepted"] != -1]
            rate     = (rated["accepted"] == 1).sum() / len(rated) if not rated.empty else 0.0
            cat_rows.append({
                "Category":        cat,
                "Total Reviews":   len(cat_df),
                "Rated":           len(rated),
                "Acceptance Rate": rate,
            })

        rate_df = pd.DataFrame(cat_rows).sort_values("Acceptance Rate", ascending=False)
        rate_df["Acceptance Rate %"] = rate_df["Acceptance Rate"].map("{:.0%}".format)
        st.dataframe(
            rate_df[["Category", "Total Reviews", "Rated", "Acceptance Rate %"]],
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            rate_df.sort_values("Acceptance Rate"),
            x="Acceptance Rate", y="Category",
            orientation="h",
            color="Acceptance Rate",
            color_continuous_scale=["#FF4444", "#FFB800", "#36A64F"],
            range_color=[0, 1],
            text=rate_df.sort_values("Acceptance Rate")["Acceptance Rate %"],
            labels={"Acceptance Rate": "Rate", "Category": ""},
        )
        fig.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # ── Convention browser ────────────────────────────────────────────────
    st.subheader("Convention Browser")
    cat_opts = ["All"] + sorted(conventions_df["category"].unique().tolist())
    sel_cat  = st.selectbox("Filter by category", cat_opts, key="conv_cat")

    display_conv = conventions_df.copy()
    if sel_cat != "All":
        display_conv = display_conv[display_conv["category"] == sel_cat]

    for _, row in display_conv.iterrows():
        lang_tag = f"`{row['language']}`" if row["language"] != "all" else "`all languages`"
        scope    = f"`{row['repo']}`"     if row["repo"]      != "global" else "`global`"
        with st.expander(
            f"**[{row['category']}]** {row['rule'][:80]}{'…' if len(row['rule']) > 80 else ''}",
            expanded=False,
        ):
            st.markdown(row["rule"])
            st.caption(f"Language: {lang_tag}  •  Scope: {scope}  •  Added by: `{row['added_by']}`")


# ── Navigation and main entry point ──────────────────────────────────────────


def main() -> None:
    """Render the selected page based on sidebar navigation."""
    st.sidebar.image(
        "https://img.icons8.com/fluency/96/bot.png",
        width=64,
    )
    st.sidebar.title("CodeReviewBot")
    st.sidebar.caption("AI-powered PR review dashboard")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigate to",
        options=[
            "📊 Overview",
            "🗂️ Repository Breakdown",
            "📜 Review History",
            "📏 Convention Effectiveness",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    if st.sidebar.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

    # Load data once — shared across all pages.
    reviews_df     = load_reviews()
    conventions_df = load_conventions()

    if page == "📊 Overview":
        page_overview(reviews_df)
    elif page == "🗂️ Repository Breakdown":
        page_repo_breakdown(reviews_df)
    elif page == "📜 Review History":
        page_review_history(reviews_df)
    elif page == "📏 Convention Effectiveness":
        page_convention_effectiveness(reviews_df, conventions_df)

    st.sidebar.divider()
    st.sidebar.caption(
        f"Data: {len(reviews_df):,} reviews · {len(conventions_df):,} conventions"
    )


if __name__ == "__main__":
    main()
