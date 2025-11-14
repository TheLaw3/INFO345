# ingest.py
"""Ingest raw ratings and items CSVs into a project data directory and emit a schema report.

This script reads the raw ratings and item metadata CSVs, makes byte‑exact
copies in a dedicated output directory (`data/raw` by default) and writes a
JSON report describing the datasets (file paths, row/column counts, column
names and simple column hints).  It accepts three command‑line arguments:
`--ratings` (path to the user–item ratings CSV), `--items` (path to the
item metadata CSV) and `--outdir` (where the copied files and report are
saved). By duplicating the raw inputs and recording their structure, the
script ensures that downstream preprocessing, analysis and modelling can
always be traced back to a fixed snapshot of the original data.

refrences 

Lectures 1 & 2 Introduction to Recommender Systems emphasise that raw
  recommender datasets are sparse and often messy; understanding dataset
  size, sparsity and column semantics is a prerequisite for building any
  model.  Copying the original ratings and item files into a stable
  directory and summarising their structure aligns with the course advice
  to document data characteristics before starting exploratory analysis.


Outputs (written under --outdir)
  books_rating_cleaned.raw.csv  Unmodified copy of the ratings source.
  books_data.raw.csv            Unmodified copy of the items source.
  ingest_report.json            Summary of shapes, columns, and soft column hints.

Libraries 
pandas: Robust CSV I/O and dtype handling for large files (used here as a dependable reader/writer).
  Alternative: polars (faster on large data, different API; not necessary for a single pass copy and report).
pathlib: Cross-platform, explicit filesystem paths. Alternative: os.path (works, but less ergonomic).
argparse/json: Standard library CLIs and structured reports. Alternatives: click/typer 

"""

import argparse, json
from pathlib import Path
import pandas as pd

def main():
    """
    Parse CLI args, copy raw CSVs to outdir, and write a schema report.
    """
    ap = argparse.ArgumentParser(description="Copy the raw data files into the project data/raw directory.")
    ap.add_argument("--ratings", required=True, help="Path to the raw ratings CSV")
    ap.add_argument("--items",   required=True, help="Path to the raw items CSV")
    ap.add_argument("--outdir",  default="data/raw")
    args = ap.parse_args()

    ratings_path = Path(args.ratings)
    items_path   = Path(args.items)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not ratings_path.exists():
        raise FileNotFoundError(f"Missing ratings file: {ratings_path}")
    if not items_path.exists():
        raise FileNotFoundError(f"Missing items file: {items_path}")

    # low_memory=False avoids mixed-type inference across chunks during read_csv,
    # which can otherwise yield inconsistent dtypes for large files.
    r = pd.read_csv(ratings_path, low_memory=False)
    m = pd.read_csv(items_path, low_memory=False)

    # Save unchanged snapshots (provenance: byte-for-byte content preserved by rewriting rows unchanged).
    r.to_csv(outdir / "books_rating_cleaned.raw.csv", index=False)
    m.to_csv(outdir / "books_data.raw.csv", index=False)

    # Write a schema report.
    # 'column_hints' are soft expectations intended to help manual inspection
    # and downstream mapping; they are not enforced here.
    report = {
        "ratings_path": str(ratings_path.resolve()),
        "items_path": str(items_path.resolve()),
        "ratings_shape": list(r.shape),
        "items_shape": list(m.shape),
        "ratings_columns": [str(c) for c in r.columns],
        "items_columns": [str(c) for c in m.columns],
        "column_hints": {
            "expected_ratings": ["user_id/User_id/reviewerID", "item_id/Id/asin", "rating or review/score"],
            "expected_items": ["item_id or Title", "Title", "categories"]
        }
    }
    (outdir / "ingest_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
