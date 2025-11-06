# prepare_data.py
import argparse
import json
from pathlib import Path

import pandas as pd

def pick(colnames, candidates):
    s = {c.lower(): c for c in colnames}
    for c in candidates:
        if c in colnames: return c
        if c.lower() in s: return s[c.lower()]
    return None

def main():
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
    r["rating"] = pd.to_numeric(r["rating"], errors="coerce").clip(1, 5)
    r = r.dropna(subset=["user_id", "item_id", "rating"])
    r["user_id"] = r["user_id"].astype(str).str.strip()
    r["item_id"] = r["item_id"].astype(str).str.strip()
    if "title" in r.columns:
        r["title"] = r["title"].astype(str).str.strip()
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
        mm["meta_title"] = mm["meta_title"].astype(str).str.strip()
        if "categories" in mm.columns:
            mm["categories"] = mm["categories"].astype(str).str.strip()
        items["t_norm"] = items["title"].astype(str).str.lower().str.strip()
        mm["t_norm"]    = mm["meta_title"].astype(str).str.lower().str.strip()
        items = items.merge(mm.drop(columns=["meta_title"]), on="t_norm", how="left")
        items = items.drop(columns=["t_norm"])

    # Text field for CBF
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

    items["text"] = items["text"].fillna("").astype(str).str.strip()
    empty = items["text"].str.len() == 0
    if empty.any():
        items.loc[empty, "text"] = items.loc[empty, "item_id"]

    # retain richest duplicate metadata rows (longest text wins)
    items = (items.assign(_text_len=items["text"].str.len())
                  .sort_values(["item_id", "_text_len"], ascending=[True, False])
                  .drop_duplicates("item_id")
                  .drop(columns="_text_len"))

    # Provide canonical Title/Categories columns if available
    if "title" in items.columns:
        items["Title"] = items["title"]
    if "categories" in items.columns:
        items["Categories"] = items["categories"]

    preferred_cols = ["item_id", "text"]
    for optional in ["Title", "Categories"]:
        if optional in items.columns:
            preferred_cols.append(optional)
    remaining = [c for c in items.columns if c not in preferred_cols]
    items = items[preferred_cols + remaining]

    # CF-ready training subset (keep full catalog for serving)
    u_sizes = r.groupby("user_id").size()
    i_sizes = r.groupby("item_id").size()
    mask = (r["user_id"].isin(u_sizes[u_sizes >= args.min_user].index) &
            r["item_id"].isin(i_sizes[i_sizes >= args.min_item].index))
    r_cf = r[mask].copy()

    # Save
    trainable_path = outdir / "trainable_ratings.csv"
    items_path = outdir / "items.csv"
    cf_path = outdir / "ratings_cf_train.csv"
    r.to_csv(trainable_path, index=False)
    items.to_csv(items_path, index=False)
    r_cf.to_csv(cf_path, index=False)

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
            "trainable_ratings": str(trainable_path.resolve()),
            "items": str(items_path.resolve()),
            "ratings_cf_train": str(cf_path.resolve()),
        }
    }
    (outdir/"preprocess_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
