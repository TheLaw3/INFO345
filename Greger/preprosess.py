# preprocess.py
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

def pick(colnames, candidates):
    s = {c.lower(): c for c in colnames}
    for c in candidates:
        if c in colnames: return c
        if c.lower() in s: return s[c.lower()]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default="data/raw/books_rating_cleaned.raw.csv")
    ap.add_argument("--items",   default="data/raw/books_data.raw.csv")
    ap.add_argument("--outdir",  default="data")
    ap.add_argument("--min_user", type=int, default=5)
    ap.add_argument("--min_item", type=int, default=5)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load
    r = pd.read_csv(args.ratings, low_memory=False)
    m = pd.read_csv(args.items, low_memory=False)

    # Map ratings columns to standard names
    rcols = list(r.columns)
    user_col = pick(rcols, ["user_id","User_id","reviewerID","user","profileName"])
    item_col = pick(rcols, ["item_id","Id","asin","ASIN","product_id","book_id","id","ID"])
    rate_col = pick(rcols, ["rating","review/score","Score","overall","stars","rate"])
    title_r  = pick(rcols, ["Title","title","name","book_title"])

    if not all([user_col, item_col, rate_col]):
        raise ValueError(f"Missing key columns. Found user:{user_col}, item:{item_col}, rating:{rate_col}")

    keep = [user_col, item_col, rate_col] + ([title_r] if title_r else [])
    r = r[keep].copy()
    r.columns = ["user_id","item_id","rating"] + (["title"] if title_r else [])

    # Clean ratings
    r["rating"] = pd.to_numeric(r["rating"], errors="coerce").clip(1,5)
    r = r.dropna(subset=["user_id","item_id","rating"])
    r["user_id"] = r["user_id"].astype(str)
    r["item_id"] = r["item_id"].astype(str)
    r = r.drop_duplicates(subset=["user_id","item_id"])

    # Build item catalog
    items = r[["item_id"]].drop_duplicates().copy()

    # Attach title from ratings if present
    if "title" in r.columns:
        t = r[["item_id","title"]].dropna().drop_duplicates("item_id")
        items = items.merge(t, on="item_id", how="left")

    # Try to add categories from meta by normalized Title
    mcols = list(m.columns)
    meta_title = pick(mcols, ["Title","title","name"])
    meta_cat   = pick(mcols, ["categories","category","genres","labels"])
    if meta_title is not None and "title" in items.columns:
        mm = m[[meta_title] + ([meta_cat] if meta_cat else [])].copy()
        mm.columns = ["meta_title"] + (["categories"] if meta_cat else [])
        items["t_norm"] = items["title"].astype(str).str.lower().str.strip()
        mm["t_norm"]    = mm["meta_title"].astype(str).str.lower().str.strip()
        items = items.merge(mm.drop(columns=["meta_title"]), on="t_norm", how="left")
        items = items.drop(columns=["t_norm"])

    # Text field for CBF
    if "title" in items.columns:
        if "categories" in items.columns:
            items["text"] = items["title"].fillna("") + " " + items["categories"].fillna("")
        else:
            items["text"] = items["title"].fillna("")

    # CF-ready training subset (keep full catalog for serving)
    u_sizes = r.groupby("user_id").size()
    i_sizes = r.groupby("item_id").size()
    mask = r["user_id"].isin(u_sizes[u_sizes>=args.min_user].index) & \
           r["item_id"].isin(i_sizes[i_sizes>=args.min_item].index)
    r_cf = r[mask].copy()

    # Save
    (outdir/"trainable_ratings.csv").write_text("")  # ensure path exists on some systems
    r.to_csv(outdir/"trainable_ratings.csv", index=False)
    items.to_csv(outdir/"items.csv", index=False)
    r_cf.to_csv(outdir/"ratings_cf_train.csv", index=False)

    # Report
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
            "trainable_ratings": str((outdir/"trainable_ratings.csv").resolve()),
            "items": str((outdir/"items.csv").resolve()),
            "ratings_cf_train": str((outdir/"ratings_cf_train.csv").resolve()),
        }
    }
    (outdir/"preprocess_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
