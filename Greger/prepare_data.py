"""Prepare ratings and item metadata for CF/CBF pipelines and emit a preprocessing report.

This script ingests raw ratings and item metadata, cleans and standardises
column names, builds a consistent item catalog with enriched text fields
for content‑based filtering, filters users and items by minimum activity
thresholds for collaborative filtering, and writes three canonical files:
(i) `trainable_ratings.csv` containing all cleaned interactions with
canonical columns (`user_id`, `item_id`, `rating`, `timestamp`);
(ii) `items.csv` containing the catalog of items with title, categories
and a concatenated `text` field for CBF models; and (iii) `ratings_cf_train.csv`
containing the subset of interactions involving users and items that
meet the `min_user` and `min_item` thresholds for CF models.  A JSON
report summarising dataset sizes, sparsity, file paths and split
parameters is also written to assist reproducibility.

refrences 

Lectures 1 & 2 – Introduction.
  Introduce the concept of sparsity in recommender datasets and explain why
  it is important to count users, items and interactions before modelling.
  The script prints these counts and the density to help understand the
  dataset’s sparsity and head/tail distributions.
  
  abel Your Data (2025), “Data Versioning: Best Practices for ML
  Engineers.”
  The article notes that a straightforward data versioning strategy is
  full duplication: saving a complete copy of the dataset whenever it
  changes; each copy acts as a snapshot:contentReference. It also
  recommends capturing metadata (schema, column names) and linking data
  versions with code and experiments, and automating versioning in
  pipelines to produce traceable snapshots:contentReference. Our
  ingestion script duplicates the raw CSVs, and this module documents the
  schema and shapes of the processed datasets in a report for
  traceability.
  URL: https://labelyourdata.com/articles/machine-learning/data-versioning 
  
  

This script:
  1) Loads raw ratings and item metadata CSVs.
  2) Maps heterogeneous column names onto canonical fields: user_id, item_id, rating, and optional title.
  3) Cleans ratings (numeric coercion, clamp to [1, 5], drop NA, strip IDs, deduplicate user-item pairs).
  4) Builds an item catalog and enriches it with optional title/categories from metadata via normalized Title match.
  5) Constructs a 'text' field per item for content-based filtering (CBF).
  6) Produces a CF-ready ratings subset using user/item activity thresholds.
  7) Saves artifacts and a JSON summary report.

Inputs:
  --ratings  Raw ratings CSV path (default: data/raw/books_rating_cleaned.raw.csv)
  --items    Raw items CSV path   (default: data/raw/books_data.raw.csv)
  --outdir   Output directory     (default: data)
  --min_user Minimum interactions per user to keep in CF subset (default: 5)
  --min_item Minimum interactions per item to keep in CF subset (default: 5)

Outputs:
  trainable_ratings.csv   Cleaned ratings with canonical columns.
  items.csv               Item catalog with text field and optional Title/Categories.
  ratings_cf_train.csv    CF-ready subset filtered by min_user/min_item.
  preprocess_report.json  Row/user/item counts and file paths.

Libraries:
  pandas: ETL, CSV I/O, grouping, joins. Chosen for readability and course alignment.
    Alternative: polars (faster; different API; not essential for this pipeline scale).
  pathlib: Cross-platform path handling. Alternative: os.path (less ergonomic).
  argparse/json: Standard library for CLIs and structured reports. Alternatives: click/typer (extra deps).

"""

import argparse
import json
from pathlib import Path

import pandas as pd

def pick(colnames, candidates):
    """
    Return the first matching column name from `colnames` given alias `candidates`.

    Matching is case-insensitive, but the return value preserves the original
    casing from `colnames`.
    """
    s = {c.lower(): c for c in colnames}
    for c in candidates:
        if c in colnames: return c
        if c.lower() in s: return s[c.lower()]
    return None

def main():
    """
    Standardize columns, clean ratings, assemble item catalog, and write artifacts.

    Creates three CSVs and a JSON report under --outdir. Filters a CF-ready
    subset based on --min_user and --min_item thresholds.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default="data/raw/books_rating_cleaned.raw.csv",
                    help="Raw ratings CSV with user/item/rating columns")
    ap.add_argument("--items",   default="data/raw/books_data.raw.csv",
                    help="Raw item metadata CSV (titles/categories optional)")
    ap.add_argument("--outdir",  default="data")
    ap.add_argument("--min_user", type=int, default=5)
    ap.add_argument("--min_item", type=int, default=5)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load source tables
    r = pd.read_csv(args.ratings, low_memory=False)
    m = pd.read_csv(args.items, low_memory=False)

    # Map ratings columns to standard names via alias picking
    rcols = list(r.columns)
    user_col = pick(rcols, ["user_id","User_id","reviewerID","user","profileName"])
    item_col = pick(rcols, ["item_id","Id","asin","ASIN","product_id","book_id","id","ID"])
    rate_col = pick(rcols, ["rating","review/score","Score","overall","stars","rate"])
    title_r  = pick(rcols, ["Title","title","name","book_title"])

    if not all([user_col, item_col, rate_col]):
        raise ValueError(f"Missing key columns. Found user:{user_col}, item:{item_col}, rating:{rate_col}")

    # Keep only the mapped columns and rename to canonical schema
    keep = [user_col, item_col, rate_col] + ([title_r] if title_r else [])
    r = r[keep].copy()
    r.columns = ["user_id","item_id","rating"] + (["title"] if title_r else [])

    # Clean ratings table
    # Coerce rating to numeric and clamp to [1, 5]
    # Drop rows with missing IDs or rating
    # Normalize IDs/titles and deduplicate user-item pairs
    r["rating"] = pd.to_numeric(r["rating"], errors="coerce").clip(1, 5)
    r = r.dropna(subset=["user_id", "item_id", "rating"])
    r["user_id"] = r["user_id"].astype(str).str.strip()
    r["item_id"] = r["item_id"].astype(str).str.strip()
    if "title" in r.columns:
        r["title"] = r["title"].astype(str).str.strip()
    r = r.drop_duplicates(subset=["user_id","item_id"])

    # Build base item catalog from observed item_ids in ratings
    items = r[["item_id"]].drop_duplicates().copy()

    # Attach title from ratings if available
    if "title" in r.columns:
        t = r[["item_id","title"]].dropna().drop_duplicates("item_id")
        items = items.merge(t, on="item_id", how="left")

    # Try to enrich with categories from metadata by normalized Title match
    mcols = list(m.columns)
    meta_title = pick(mcols, ["Title","title","name"])
    meta_cat   = pick(mcols, ["categories","category","genres","labels"])
    if meta_title is not None and "title" in items.columns:
        mm = m[[meta_title] + ([meta_cat] if meta_cat else [])].copy()
        mm.columns = ["meta_title"] + (["categories"] if meta_cat else [])
        mm["meta_title"] = mm["meta_title"].astype(str).str.strip()
        if "categories" in mm.columns:
            mm["categories"] = mm["categories"].astype(str).str.strip()
        # Normalize titles for a case-insensitive, whitespace-tolerant join
        items["t_norm"] = items["title"].astype(str).str.lower().str.strip()
        mm["t_norm"]    = mm["meta_title"].astype(str).str.lower().str.strip()
        items = items.merge(mm.drop(columns=["meta_title"]), on="t_norm", how="left")
        items = items.drop(columns=["t_norm"])

    # Construct a CBF text field using available title/categories
    if "title" in items.columns:
        if "categories" in items.columns:
            items["text"] = (items["title"].fillna("") + " " +
                              items["categories"].fillna("")).str.strip()
        else:
            items["text"] = items["title"].fillna("").str.strip()
    elif "categories" in items.columns:
        items["text"] = items["categories"].fillna("").str.strip()
    else:
        items["text"] = ""

    # Guarantee non-empty text; fallback to item_id, this ensures CBF never sees empty strings
    items["text"] = items["text"].fillna("").astype(str).str.strip()
    empty = items["text"].str.len() == 0
    if empty.any():
        items.loc[empty, "text"] = items.loc[empty, "item_id"]

    # Retain richest duplicate metadata rows per item_id (longest text wins)
    items = (items.assign(_text_len=items["text"].str.len())
                  .sort_values(["item_id", "_text_len"], ascending=[True, False])
                  .drop_duplicates("item_id")
                  .drop(columns="_text_len"))

    # Provide canonical Title/Categories columns if available for downstream tools
    if "title" in items.columns:
        items["Title"] = items["title"]
    if "categories" in items.columns:
        items["Categories"] = items["categories"]

    # Column ordering: keep "item_id","text" first, 
    preferred_cols = ["item_id", "text"]
    for optional in ["Title", "Categories"]:
        if optional in items.columns:
            preferred_cols.append(optional)
    remaining = [c for c in items.columns if c not in preferred_cols]
    items = items[preferred_cols + remaining]

    # CF-ready subset: filter by user/item activity thresholds
    u_sizes = r.groupby("user_id").size()
    i_sizes = r.groupby("item_id").size()
    mask = (r["user_id"].isin(u_sizes[u_sizes >= args.min_user].index) &
            r["item_id"].isin(i_sizes[i_sizes >= args.min_item].index))
    r_cf = r[mask].copy()

    # Save artifacts
    trainable_path = outdir / "trainable_ratings.csv"
    items_path = outdir / "items.csv"
    cf_path = outdir / "ratings_cf_train.csv"
    r.to_csv(trainable_path, index=False)
    items.to_csv(items_path, index=False)
    r_cf.to_csv(cf_path, index=False)

    # Emit a preprocessing report 
    report = {
        "rows_base": int(len(r)),
        "users_base": int(r["user_id"].nunique()),
        "items_base": int(r["item_id"].nunique()),
        "rows_cf": int(len(r_cf)),
        "users_cf_min_user": int(r_cf["user_id"].nunique()),
        "items_cf_min_item": int(r_cf["item_id"].nunique()),
        "min_user": args.min_user,
        "min_item": args.min_item,
        "paths": {
            "trainable_ratings": str(trainable_path.resolve()),
            "items": str(items_path.resolve()),
            "ratings_cf_train": str(cf_path.resolve()),
        }
    }
    (outdir/"preprocess_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
