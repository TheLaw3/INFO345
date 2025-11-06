# ingest.py
import argparse, json
from pathlib import Path
import pandas as pd

def main():
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

    r = pd.read_csv(ratings_path, low_memory=False)
    m = pd.read_csv(items_path, low_memory=False)

    # Save unchanged snapshots
    r.to_csv(outdir / "books_rating_cleaned.raw.csv", index=False)
    m.to_csv(outdir / "books_data.raw.csv", index=False)

    # Write a schema report
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
