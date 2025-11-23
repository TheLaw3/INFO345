
""""
Src/eda_plots.py — generate EDA plots for the standardized data.

   This script generates various exploratory data analysis (EDA) plots
   for the training data used in our recommender models. It reads a ratings
   CSV (user_id, item_id, rating) and, if available, an items CSV containing
   item metadata. The script creates and saves plots visualizing the
   distribution of ratings, user activity levels, item popularity, and the
   long-tail curve of item interactions. The plots are saved in PNG and PDF
   formats in the specified output directory. 
   
   look at Src/eda.py for more details on the dataset and EDA context.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


RATINGS_FILE = Path("data/trainable_ratings.csv")  # after prepare_data.py
ITEMS_FILE = Path("data/items.csv")                
OUT_DIR = Path("figures")

USER_COL = "user_id"
ITEM_COL = "item_id"
RATING_COL = "rating"


plt.style.use("seaborn-v0_8-whitegrid")


def load_data():
    if not RATINGS_FILE.exists():
        raise FileNotFoundError(f"Ratings file not found: {RATINGS_FILE}")

    print(f"Loading ratings from {RATINGS_FILE} ...")
    ratings = pd.read_csv(RATINGS_FILE)

    items = None
    if ITEMS_FILE.exists():
        print(f"Loading items from {ITEMS_FILE} ...")
        items = pd.read_csv(ITEMS_FILE)
    else:
        print(f"Items file not found ({ITEMS_FILE}), continuing without it.")

    # Basic sanity print
    print(f"#rows in ratings: {len(ratings)}")
    print(f"Columns in ratings: {list(ratings.columns)}")
    return ratings, items


def plot_rating_distribution(ratings: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))

    values = ratings[RATING_COL].dropna()
    bins = np.arange(values.min() - 0.5, values.max() + 1.5, 1.0)
    ax.hist(values, bins=bins, edgecolor="black", align="mid")

    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Ratings (Amazon Books)")
    ax.set_xticks(sorted(values.unique()))

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"rating_distribution.{ext}", dpi=300)
    plt.close(fig)
    print("Saved rating_distribution.{png,pdf}")


def plot_user_activity_distribution(ratings: pd.DataFrame, out_dir: Path):
    user_counts = (
        ratings.groupby(USER_COL)[ITEM_COL]
        .size()
        .rename("n_interactions")
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(user_counts, bins=50, edgecolor="black")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("# interactions per user (log scale)")
    ax.set_ylabel("# users (log scale)")
    ax.set_title("User Activity Distribution")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"user_activity_distribution.{ext}", dpi=300)
    plt.close(fig)
    print("Saved user_activity_distribution.{png,pdf}")

    print(
        f"User activity: min={user_counts.min()}, "
        f"median={user_counts.median()}, "
        f"max={user_counts.max()}"
    )


def plot_item_popularity_distribution(ratings: pd.DataFrame, out_dir: Path):
    item_counts = (
        ratings.groupby(ITEM_COL)[USER_COL]
        .size()
        .rename("n_interactions")
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(item_counts, bins=50, edgecolor="black")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("# interactions per item (log scale)")
    ax.set_ylabel("# items (log scale)")
    ax.set_title("Item Popularity Distribution")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"item_popularity_distribution.{ext}", dpi=300)
    plt.close(fig)
    print("Saved item_popularity_distribution.{png,pdf}")

    print(
        f"Item popularity: min={item_counts.min()}, "
        f"median={item_counts.median()}, "
        f"max={item_counts.max()}"
    )


def plot_long_tail_curve(ratings: pd.DataFrame, out_dir: Path):
    item_counts = (
        ratings.groupby(ITEM_COL)[USER_COL]
        .size()
        .rename("n_interactions")
        .sort_values(ascending=False)
    )

    n_items = len(item_counts)
    ranks = np.arange(1, n_items + 1)
    frac_items = ranks / n_items
    cum_interactions = item_counts.cumsum() / item_counts.sum()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(frac_items, cum_interactions, lw=2)

    # Mark top 20% most popular items
    head_cutoff = 0.2
    head_index = int(np.floor(head_cutoff * n_items))
    head_share = cum_interactions.iloc[head_index - 1]

    ax.axvline(head_cutoff, linestyle="--", label="Top 20% items")
    ax.axhline(head_share, linestyle="--")
    ax.scatter([head_cutoff], [head_share])

    ax.set_xlabel("Fraction of items (sorted by popularity)")
    ax.set_ylabel("Fraction of all interactions (cumulative)")
    ax.set_title("Long-Tail Curve (Item Popularity)")

    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"long_tail_curve.{ext}", dpi=300)
    plt.close(fig)
    print("Saved long_tail_curve.{png,pdf}")

    print(
        f"Top 20% most popular items account for "
        f"{head_share * 100:.1f}% of all interactions."
    )


def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    ratings, items = load_data()


    n_users = ratings[USER_COL].nunique()
    n_items = ratings[ITEM_COL].nunique()
    n_ratings = len(ratings)
    sparsity = 1 - n_ratings / (n_users * n_items)

    print("\n=== Basic dataset stats (after filtering) ===")
    print(f"#users      = {n_users}")
    print(f"#items      = {n_items}")
    print(f"#ratings    = {n_ratings}")
    print(f"sparsity    = {sparsity:.4f} (1 - |R| / (|U|*|I|))")

    # Generate plots
    plot_rating_distribution(ratings, OUT_DIR)
    plot_user_activity_distribution(ratings, OUT_DIR)
    plot_item_popularity_distribution(ratings, OUT_DIR)
    plot_long_tail_curve(ratings, OUT_DIR)


if __name__ == "__main__":
    main()
