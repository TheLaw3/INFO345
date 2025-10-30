# split.py
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

TIME_CANDIDATES = ["timestamp","time","unixReviewTime","reviewTime","date"]

def pick(colnames, candidates):
    s = {c.lower(): c for c in colnames}
    for c in candidates:
        if c in colnames: return c
        if c.lower() in s: return s[c.lower()]
    return None

def split_user_random(dfu, rng):
    n = len(dfu)
    if n == 1:
        return dfu, None, None
    idx = dfu.index.tolist()
    test_i = rng.choice(idx)
    idx.remove(test_i)
    val_i = rng.choice(idx) if n >= 3 else None
    train = dfu.loc[idx]
    val   = dfu.loc[[val_i]] if val_i is not None else None
    test  = dfu.loc[[test_i]]
    return train, val, test

def split_user_temporal(dfu, time_col):
    dfu = dfu.sort_values(time_col)
    n = len(dfu)
    if n == 1:
        return dfu, None, None
    test  = dfu.tail(1)
    remain = dfu.iloc[:-1]
    val   = remain.tail(1) if n >= 3 else None
    train = remain.iloc[:-1] if n >= 3 else remain
    return train, val, test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", required=True, help="cleaned ratings CSV with user_id,item_id,rating")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_train_inter", type=int, default=1, help="drop users with fewer train rows")
    ap.add_argument("--strategy", choices=["random","temporal"], default="random")
    ap.add_argument("--time_col", default="", help="optional explicit timestamp column")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    r = pd.read_csv(args.ratings, low_memory=False)

    need = {"user_id","item_id","rating"}
    if not need.issubset(r.columns):
        raise ValueError(f"Missing columns: need {need}, have {set(r.columns)}")

    # ensure types and single opinion per pair
    r["rating"] = pd.to_numeric(r["rating"], errors="coerce").clip(1,5)
    r = r.dropna(subset=["user_id","item_id","rating"])
    r["user_id"] = r["user_id"].astype(str)
    r["item_id"] = r["item_id"].astype(str)
    r = r.drop_duplicates(subset=["user_id","item_id"])

    # pick time column if temporal
    time_col = None
    if args.strategy == "temporal":
        time_col = args.time_col or pick(list(r.columns), TIME_CANDIDATES)
        if time_col is None:
            raise ValueError("No timestamp column found; supply --time_col or use --strategy random")

    rng = np.random.default_rng(args.seed)
    trains, vals, tests = [], [], []

    for uid, dfu in r.groupby("user_id", sort=False):
        if args.strategy == "temporal":
            tr, va, te = split_user_temporal(dfu, time_col)
        else:
            tr, va, te = split_user_random(dfu, rng)
        trains.append(tr)
        if va is not None and len(va): vals.append(va)
        if te is not None and len(te): tests.append(te)

    train = pd.concat(trains, ignore_index=True)
    val   = pd.concat(vals,   ignore_index=True) if len(vals)   else pd.DataFrame(columns=r.columns)
    test  = pd.concat(tests,  ignore_index=True) if len(tests)  else pd.DataFrame(columns=r.columns)

    # enforce minimum train interactions per user
    if args.min_train_inter > 1:
        keep = train["user_id"].value_counts()
        keep = set(keep[keep >= args.min_train_inter].index)
        train = train[train["user_id"].isin(keep)]
        val   = val[val["user_id"].isin(keep)]
        test  = test[test["user_id"].isin(keep)]

    # write splits
    train.to_csv(outdir/"train.csv", index=False)
    val.to_csv(outdir/"val.csv", index=False)
    test.to_csv(outdir/"test.csv", index=False)

    # stats
    stats = {
        "seed": args.seed,
        "strategy": args.strategy,
        "time_col": time_col,
        "n_rows_in": int(len(r)),
        "n_users_in": int(r["user_id"].nunique()),
        "n_items_in": int(r["item_id"].nunique()),
        "n_rows_train": int(len(train)),
        "n_rows_val": int(len(val)),
        "n_rows_test": int(len(test)),
        "n_users_train": int(train["user_id"].nunique()),
        "n_users_val": int(val["user_id"].nunique()),
        "n_users_test": int(test["user_id"].nunique()),
        "n_items_train": int(train["item_id"].nunique()),
        "n_items_val": int(val["item_id"].nunique()),
        "n_items_test": int(test["item_id"].nunique()),
        "min_train_inter": args.min_train_inter
    }
    (outdir/"split_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
