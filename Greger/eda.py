"""Greger/eda.py — quick exploratory analysis for the standardized data."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_ratings(path: Path) -> pd.DataFrame:
    """Load and clean a standardized ratings CSV.

    Ensures presence of columns {user_id, item_id, rating}, coerces rating to
    numeric in [1, 5], strips identifiers, and drops invalid rows.

    Args:
      path: Path to a CSV file with columns user_id, item_id, rating.

    Returns:
      A cleaned pandas DataFrame with columns user_id, item_id, rating.

    Raises:
      ValueError: If required columns are missing or all rows become invalid.
    """
    df = pd.read_csv(path, low_memory=False)
    required = {"user_id", "item_id", "rating"}
    if not required.issubset(df.columns):
        raise ValueError(f"Ratings file {path} is missing columns: {required - set(df.columns)}")
    df = df.dropna(subset=["user_id", "item_id", "rating"]).copy()
    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["item_id"] = df["item_id"].astype(str).str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").clip(1, 5)
    df = df.dropna(subset=["rating"])
    if df.empty:
        raise ValueError(f"Ratings file {path} has no valid rows after cleaning")
    return df


def attach_titles(ratings: pd.DataFrame, items_path: Path) -> pd.DataFrame:
    """Merge human-readable item metadata into the ratings frame if available.

    Joins on item_id using an items CSV and carries over common metadata fields
    such as Title/title, Categories/categories, and text.

    Args:
      ratings: Cleaned ratings DataFrame with column item_id.
      items_path: Path to items CSV. If missing or malformed, ratings are returned unchanged.

    Returns:
      Ratings DataFrame with additional metadata columns when present.
    """
    if not items_path.exists():
        return ratings
    items = pd.read_csv(items_path, low_memory=False)
    if "item_id" not in items.columns:
        return ratings
    cols = ["item_id"] + [c for c in ["Title", "title", "Categories", "categories", "text"] if c in items.columns]
    items = items[cols].copy()
    items["item_id"] = items["item_id"].astype(str).str.strip()
    for c in items.columns:
        if c != "item_id":
            items[c] = items[c].astype(str).str.strip()
    merged = ratings.merge(items, on="item_id", how="left")
    return merged


def hist(ax, series, bins, title, xlabel):
    """Plot a linear-scale histogram on the provided axes.

    Args:
      ax: Matplotlib Axes to draw on.
      series: 1D array-like of numeric values.
      bins: Number of bins or bin edges.
      title: Plot title.
      xlabel: X-axis label.
    """
    ax.hist(series, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")


def log_hist(ax, series, bins, title, xlabel):
    """Plot a histogram with log-scaled axes.

    Args:
      ax: Matplotlib Axes to draw on.
      series: 1D array-like of numeric values.
      bins: Number of bins or bin edges (use log-spaced for readability).
      title: Plot title.
      xlabel: X-axis label.
    """
    ax.hist(series, bins=bins)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")


def main() -> None:
    """Generate summary stats and optional histograms for standardized ratings.

    CLI:
      --ratings     Path to trainable_ratings.csv (default: data/trainable_ratings.csv)
      --items       Path to items.csv for title/categories merge (default: data/items.csv)
      --outdir      Output directory for reports and plots (default: results/eda)
      --save-plots  If set, save PNG histograms under <outdir>/plots

    Outputs:
      - <outdir>/eda_stats.json with core dataset statistics.
      - Optional PNGs: ratings_hist, user_activity_hist, item_popularity_hist,
        user_activity_loglog, item_popularity_loglog.

    Raises:
      ValueError: If ratings are missing required columns or contain no valid rows after cleaning.
    """
    ap = argparse.ArgumentParser(description="Generate quick EDA summaries for standardized ratings data.")
    ap.add_argument("--ratings", default="data/trainable_ratings.csv")
    ap.add_argument("--items", default="data/items.csv")
    ap.add_argument("--outdir", default="results/eda")
    ap.add_argument("--save-plots", action="store_true", help="Save histogram PNGs instead of skipping plots")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ratings = load_ratings(Path(args.ratings))
    ratings = attach_titles(ratings, Path(args.items))

    n_users = int(ratings["user_id"].nunique())
    n_items = int(ratings["item_id"].nunique())
    n_inter = int(len(ratings))
    density = float(n_inter / max(1, n_users * n_items))

    user_activity = ratings.groupby("user_id").size()
    item_pop = ratings.groupby("item_id").size()

    rating_summary = ratings["rating"].describe().to_dict()
    rating_shares = (
        ratings["rating"].value_counts(normalize=True).sort_index().round(6).to_dict()
    )

    summary = {
        "ratings_path": str(Path(args.ratings).resolve()),
        "items_path": str(Path(args.items).resolve()),
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_inter,
        "density": density,
        "rating_summary": rating_summary,
        "rating_shares": rating_shares,
        "users_with_single_rating_pct": float(user_activity.eq(1).mean()),
        "items_with_single_rating_pct": float(item_pop.eq(1).mean()),
    }

    if "Title" in ratings.columns or "title" in ratings.columns:
        title_col = "Title" if "Title" in ratings.columns else "title"
        summary["missing_titles_pct"] = float(ratings[title_col].isna().mean())

    if len(item_pop):
        pop_sorted = item_pop.sort_values(ascending=False).values
        cut = max(1, int(0.20 * len(pop_sorted)))
        summary["top20pct_popularity_share"] = float(pop_sorted[:cut].sum() / pop_sorted.sum())

    stats_path = outdir / "eda_stats.json"
    stats_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if args.save_plots:
        plots_dir = outdir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots()
        hist(ax, ratings["rating"], bins=20, title="Rating distribution", xlabel="rating")
        fig.tight_layout()
        fig.savefig(plots_dir / "ratings_hist.png")
        plt.close(fig)

        fig, ax = plt.subplots()
        hist(ax, user_activity, bins=50, title="User activity", xlabel="ratings per user")
        fig.tight_layout()
        fig.savefig(plots_dir / "user_activity_hist.png")
        plt.close(fig)

        fig, ax = plt.subplots()
        hist(ax, item_pop, bins=50, title="Item popularity", xlabel="ratings per item")
        fig.tight_layout()
        fig.savefig(plots_dir / "item_popularity_hist.png")
        plt.close(fig)

        if len(user_activity):
            fig, ax = plt.subplots()
            max_user = max(1, user_activity.max())
            bins_u = np.logspace(0, np.log10(max_user), 50)  # log-spaced bins for readability
            log_hist(ax, user_activity, bins=bins_u, title="User activity (log-log)", xlabel="#ratings per user")
            fig.tight_layout()
            fig.savefig(plots_dir / "user_activity_loglog.png")
            plt.close(fig)

        if len(item_pop):
            fig, ax = plt.subplots()
            max_item = max(1, item_pop.max())
            bins_i = np.logspace(0, np.log10(max_item), 50)  # log-spaced bins for readability
            log_hist(ax, item_pop, bins=bins_i, title="Item popularity (log-log)", xlabel="#ratings per item")
            fig.tight_layout()
            fig.savefig(plots_dir / "item_popularity_loglog.png")
            plt.close(fig)


if __name__ == "__main__":
    main()
