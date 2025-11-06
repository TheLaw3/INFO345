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

def pick_idx_prefer_supported(dfu, rng, global_cnt, remaining_support, min_support, forbid=None):
    """Pick a row index, preferring items that keep catalog coverage intact."""
    forbid = set() if forbid is None else set(forbid)
    candidates = [i for i in dfu.index if i not in forbid]
    if not candidates:
        return None

    def want(idx):
        iid = str(dfu.at[idx, "item_id"])
        cnt = global_cnt.get(iid, 0)
        rem = remaining_support.get(iid, cnt)
        return cnt >= min_support and rem > 1

    prefer = [i for i in candidates if want(i)]
    if prefer:
        return rng.choice(prefer)

    fallback_supported = [i for i in candidates if global_cnt.get(str(dfu.at[i, "item_id"]), 0) >= min_support]
    if fallback_supported:
        return rng.choice(fallback_supported)

    safe = [i for i in candidates if remaining_support.get(str(dfu.at[i, "item_id"]), 1) > 1]
    if safe:
        return rng.choice(safe)

    return rng.choice(candidates)

def split_user_random(dfu, rng, global_cnt, remaining_support, min_support):
    n = len(dfu)
    if n == 1:
        return dfu, None, None

    test_i = pick_idx_prefer_supported(dfu, rng, global_cnt, remaining_support, min_support)
    if test_i is None:
        raise RuntimeError("No candidate found for test split")
    forbid = {test_i}
    iid_test = str(dfu.at[test_i, "item_id"])
    remaining_support[iid_test] = max(0, remaining_support.get(iid_test, global_cnt.get(iid_test, 0)) - 1)
    val_i = None
    if n >= 3:
        val_i = pick_idx_prefer_supported(dfu, rng, global_cnt, remaining_support, min_support, forbid=forbid)
        if val_i is not None:
            forbid.add(val_i)
            iid_val = str(dfu.at[val_i, "item_id"])
            remaining_support[iid_val] = max(0, remaining_support.get(iid_val, global_cnt.get(iid_val, 0)) - 1)

    train_idx = [i for i in dfu.index if i not in forbid]
    train = dfu.loc[train_idx]
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
    ap.add_argument("--min_item_support", type=int, default=2,
                    help="prefer placing validation/test samples on items with at least this many total interactions")
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
    r["user_id"] = r["user_id"].astype(str).str.strip()
    r["item_id"] = r["item_id"].astype(str).str.strip()
    r = r.drop_duplicates(subset=["user_id","item_id"])

    # pick time column if temporal
    time_col = None
    if args.strategy == "temporal":
        time_col = args.time_col or pick(list(r.columns), TIME_CANDIDATES)
        if time_col is None:
            raise ValueError("No timestamp column found; supply --time_col or use --strategy random")

    rng = np.random.default_rng(args.seed)
    trains, vals, tests = [], [], []

    global_cnt = r["item_id"].astype(str).value_counts().to_dict()
    remaining_support = dict(global_cnt)

    for uid, dfu in r.groupby("user_id", sort=False):
        if args.strategy == "temporal":
            tr, va, te = split_user_temporal(dfu, time_col)
        else:
            tr, va, te = split_user_random(dfu, rng, global_cnt, remaining_support, args.min_item_support)
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
        "min_train_inter": args.min_train_inter,
        "min_item_support": args.min_item_support,
        "val_items_covered_in_train": float(val["item_id"].isin(train["item_id"]).mean()) if len(val) else None,
        "test_items_covered_in_train": float(test["item_id"].isin(train["item_id"]).mean()) if len(test) else None
    }
    (outdir/"split_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
