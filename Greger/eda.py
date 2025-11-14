"""Greger/eda.py — quick exploratory analysis for the standardized data.

  This script performs a quick, deterministic EDA on the training data used
for our recommender models.  It reads a ratings CSV (user_id, item_id,
rating) and, optionally, an items CSV containing item metadata, cleans and
merges the data, computes summary statistics, and optionally generates
histograms to visualize rating distributions, user activity and item
popularity.  The results are written to a JSON file and, if requested,
PNG plots are saved.  The goal is to understand dataset size, sparsity,
distributional properties and potential cold‑start issues before building
recommendation models.

refrences 

Lectures 1 & 2 – Introduction to Recommender Systems.  
  The introductory slides emphasise that recommendation datasets are
  typically very sparse (only a tiny fraction of the user–item matrix is
  filled) and exhibit a long‑tail distribution of user activity and item
  popularity.  Calculating `n_users`, `n_items`, `n_interactions` and
  density (`|R| / (|U|·|I|)`) helps us gauge sparsity and anticipate
  challenges for CF/CBF models.  Plotting histograms of user activity and
  item popularity (both linear and log–log) reveals the head–tail
  imbalance discussed in the lectures.  The percentage of users/items
  with a single rating and the top‑20% popularity share quantify coldness
  and head concentration.
  
  analytics Vidhya, “Step‑by‑step Exploratory Data Analysis using Python.” 
   This article defines EDA as the process of performing initial
   investigations on data to discover patterns and check assumptions with
   summary statistics and graphical representations; EDA can be leveraged
   to identify outliers, patterns, trends and clues for imputing missing
   values. It notes that statistics summaries
   (count, mean, std, etc.) help identify outliers and skewness.
   We follow this guidance by computing `rating_summary`, `rating_shares`
   and percentages of users/items with a single interaction, and by using
   histograms to visualize distributions.
   URL: https://www.analyticsvidhya.com/blog/2022/07/step-by-step-exploratory-data-analysis-eda-using-python/

Shruti Udupa (2025), “First Try at Building a Recommendation System:
   Exploratory Data Analysis.” 
   In the EDA phase of a recommender project, Udupa emphasises three
   stages: (i) initial cleaning of datasets, treating missing values and
   identifying outliers; (ii) in‑depth analysis and visualization of
   movie features, user behaviour patterns and rating distributions to
   uncover relationships; and (iii) engineering new predictive features or
   removing uninformative ones based on these insights.
   Our script mirrors this workflow: `load_ratings` cleans and coerces
   ratings, dropping invalid rows; `attach_titles` merges item metadata to
   enrich the data; the core analysis computes descriptive statistics and
   head–tail measures; optional plots visualize rating, user and item
   distributions; and the resulting statistics (e.g. density, cold‑user
   percentages, popularity share) inform subsequent modelling choices.
   URL: https://medium.com/pythoneers/first-try-at-building-an-end-to-end-recommendation-system-exploratory-data-analysis-c90cfd1b6ad6


Inputs/Outputs
  Inputs:
    - --ratings: CSV with columns {user_id, item_id, rating}.
    - --items:   items CSV with {item_id, Title/title, Categories/categories, text}.
  Outputs:
    - <outdir>/eda_stats.json          Dataset summary for reports and sanity checks.

What we measure and why
  n_users, n_items, n_interactions: dataset size and shape.
  density = |R| / (|U|·|I|): sparsity level, key for CF/CBF feasibility.
  rating_summary, rating_shares: label distribution; informs threshold=4.0 choice.
  users_with_single_rating_pct, items_with_single_rating_pct: coldness indicators.
  top20pct_popularity_share: head concentration; affects popularity baselines and novelty.

Libraries
  pandas: mature CSV/series ops. Alternative: polars (faster) but adds friction for course baselines.
  numpy: basic numerics and array ops; lightweight and ubiquitous.
  matplotlib: direct control for static PNGs; stable in headless mode with Agg.
    Alternatives: seaborn/plotnine (higher-level styling) not required for minimal, reproducible plots.
  pathlib/json/argparse: stdlib for paths, reports, and CLI; no extra deps.

"""

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
    """
    Plot a linear-scale histogram on the provided axes.
    """
    ax.hist(series, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")


def log_hist(ax, series, bins, title, xlabel):
    """
    Plot a histogram with log-scaled axes.
    """
    ax.hist(series, bins=bins)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")


def main() -> None:
    """
    Generate summary stats and histograms for standardized ratings.

    CLI:
      --ratings     Path to trainable_ratings.csv (default: data/trainable_ratings.csv)
      --items       Path to items.csv for title/categories merge (default: data/items.csv)
      --outdir      Output directory for reports and plots (default: results/eda)
      --save-plots  If set, save PNG histograms under <outdir>/plots

    Outputs:
      - <outdir>/eda_stats.json with core dataset statistics.
      - PNGs: ratings_hist, user_activity_hist, item_popularity_hist,
        user_activity_loglog, item_popularity_loglog.

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
            bins_u = np.logspace(0, np.log10(max_user), 50)  # log-spaced bins reveal head–tail spread
            log_hist(ax, user_activity, bins=bins_u, title="User activity (log-log)", xlabel="#ratings per user")
            fig.tight_layout()
            fig.savefig(plots_dir / "user_activity_loglog.png")
            plt.close(fig)

        if len(item_pop):
            fig, ax = plt.subplots()
            max_item = max(1, item_pop.max())
            bins_i = np.logspace(0, np.log10(max_item), 50)  # log-spaced bins reveal head–tail spread
            log_hist(ax, item_pop, bins=bins_i, title="Item popularity (log-log)", xlabel="#ratings per item")
            fig.tight_layout()
            fig.savefig(plots_dir / "item_popularity_loglog.png")
            plt.close(fig)


if __name__ == "__main__":
    main()
