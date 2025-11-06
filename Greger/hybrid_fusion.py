# Greger/hybrid_fusion.py — late-fusion hybrid for Top-K
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

#  metrics 
def ndcg_at_k(rec_items, rel_set, k):
    if k == 0: return 0.0
    dcg = 0.0
    for rank, iid in enumerate(rec_items[:k], start=1):
        if iid in rel_set: dcg += 1.0 / math.log2(rank + 1)
    ideal = min(k, len(rel_set))
    if ideal == 0: return 0.0
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal + 1))
    return dcg / idcg

def precision_at_k(rec_items, rel_set, k):
    if k == 0: return 0.0
    return sum(i in rel_set for i in rec_items[:k]) / k

def recall_at_k(rec_items, rel_set, k):
    if not rel_set: return np.nan
    return sum(i in rel_set for i in rec_items[:k]) / len(rel_set)

def hitrate_at_k(rec_items, rel_set, k):
    return 1.0 if any(i in rel_set for i in rec_items[:k]) else 0.0

def eval_topk(recs_df, eval_df, k):
    rel_per_user = eval_df.groupby("user_id")["item_id"].apply(set)
    recs_k = recs_df[recs_df["rank"] <= k]
    got = recs_k.groupby("user_id")["item_id"].apply(list)
    users = sorted(rel_per_user.index.intersection(got.index))
    precs, recs, ndcgs, hits = [], [], [], []
    for u in users:
        rel, rec = rel_per_user[u], got[u]
        precs.append(precision_at_k(rec, rel, k))
        r = recall_at_k(rec, rel, k)
        if not np.isnan(r): recs.append(r)
        ndcgs.append(ndcg_at_k(rec, rel, k))
        hits.append(hitrate_at_k(rec, rel, k))
    return {
        "users_evaluated": int(len(users)),
        f"precision@{k}": float(np.mean(precs)) if precs else 0.0,
        f"recall@{k}":    float(np.mean(recs))  if recs  else 0.0,
        f"ndcg@{k}":      float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"hit_rate@{k}":  float(np.mean(hits))  if hits  else 0.0,
    }

#  helpers 
def load_recs(path, src_name):
    df = pd.read_csv(path)
    # keep only needed cols
    keep = [c for c in ["user_id","item_id","score","rank"] if c in df.columns]
    df = df[keep].copy()
    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["item_id"] = df["item_id"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["user_id","item_id"], keep="first")
    if "score" not in df.columns:
        # if only rank present, invert rank as a proxy score
        df["score"] = -df["rank"].astype(float)
    df.rename(columns={"score": f"{src_name}_score"}, inplace=True)
    return df

def add_user_z(df, score_col, out_col):
    if len(df) == 0:
        df[out_col] = []
        return df
    g = df.groupby("user_id")[score_col]
    mu  = g.transform("mean")
    std = g.transform("std").replace(0, 1.0)
    df[out_col] = (df[score_col] - mu) / std
    return df

def build_eval(df_path, thr):
    df = pd.read_csv(df_path)
    df = df.dropna(subset=["user_id","item_id"]).copy()
    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["item_id"] = df["item_id"].astype(str).str.strip()
    df["rating"]  = pd.to_numeric(df["rating"], errors="coerce").clip(1,5)
    df = df.dropna(subset=["rating"])
    df = df.drop_duplicates(subset=["user_id","item_id"], keep="last")
    return df[df["rating"] >= thr][["user_id","item_id","rating"]]

def item_pop_from_train(train_path):
    tr = pd.read_csv(train_path)
    tr = tr.dropna(subset=["item_id"]).copy()
    tr["item_id"] = tr["item_id"].astype(str).str.strip()
    pop = tr.groupby("item_id").size().astype(float)
    # log-scale then global z-norm
    lp = np.log1p(pop)
    z = (lp - lp.mean()) / (lp.std() if lp.std() > 0 else 1.0)
    return z  # pd.Series indexed by item_id

def fuse_one_split(cf_path, cbf_path, eval_gt, K, w_cf, w_cbf, w_pop, pop_z=None, limit_users=None, out_csv=None):
    cf  = load_recs(cf_path,  "cf")
    cbf = load_recs(cbf_path, "cbf")
    # per-user z-scores
    cf  = add_user_z(cf,  "cf_score",  "cf_z")
    cbf = add_user_z(cbf, "cbf_score", "cbf_z")
    # outer-merge union of items per user
    merged = cf.merge(cbf, on=["user_id","item_id"], how="outer")
    merged[["cf_z","cbf_z"]] = merged[["cf_z","cbf_z"]].fillna(0.0)

    if pop_z is not None and w_pop != 0.0:
        merged["pop_z"] = merged["item_id"].map(pop_z).fillna(0.0)
    else:
        merged["pop_z"] = 0.0

    merged["hybrid_score"] = w_cf*merged["cf_z"] + w_cbf*merged["cbf_z"] + w_pop*merged["pop_z"]

    # keep only users in eval set
    eval_users = eval_gt["user_id"].unique()
    merged = merged[merged["user_id"].isin(eval_users)]

    if limit_users:
        keep = set(eval_users[:int(limit_users)])
        merged = merged[merged["user_id"].isin(keep)]

    # rank per user
    merged.sort_values(["user_id","hybrid_score"], ascending=[True, False], inplace=True)
    merged["rank"] = merged.groupby("user_id").cumcount() + 1
    recs = merged[["user_id","item_id","rank","hybrid_score"]]
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        recs.to_csv(out_csv, index=False)

    # metric
    metrics = eval_topk(recs, eval_gt, K)
    return recs, metrics

#  main 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf_val",  required=True)
    ap.add_argument("--cf_test", required=True)
    ap.add_argument("--cbf_val",  required=True)
    ap.add_argument("--cbf_test", required=True)
    ap.add_argument("--val",   required=True, help="ground-truth val split CSV")
    ap.add_argument("--test",  required=True, help="ground-truth test split CSV")
    ap.add_argument("--train", default="", help="train CSV for popularity tie-breaker")
    ap.add_argument("--k_top", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=4.0)
    ap.add_argument("--w_cf",  type=float, default=0.7)
    ap.add_argument("--w_cbf", type=float, default=0.3)
    ap.add_argument("--w_pop", type=float, default=0.0)
    ap.add_argument("--tune_weights", action="store_true",
                    help="grid-search fusion weights on the validation split")
    ap.add_argument("--grid_cf", default="0.6,0.7,0.8,0.9",
                    help="comma separated candidate weights for CF when tuning")
    ap.add_argument("--grid_cbf", default="0.1,0.2,0.3,0.4",
                    help="comma separated candidate weights for CBF when tuning")
    ap.add_argument("--grid_pop", default="0.0,0.05,0.1",
                    help="comma separated candidate weights for popularity when tuning")
    ap.add_argument("--tune_metric", default="ndcg",
                    choices=["precision", "recall", "ndcg", "hit_rate"],
                    help="validation metric to maximise during tuning")
    ap.add_argument("--limit_users_val",  type=int, default=None)
    ap.add_argument("--limit_users_test", type=int, default=None)
    ap.add_argument("--outdir", default="out/hybrid")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    if args.w_cf == 0 and args.w_cbf == 0 and args.w_pop == 0:
        raise ValueError("all weights are zero")

    pop_z = item_pop_from_train(args.train) if (args.train and Path(args.train).exists() and args.w_pop != 0.0) else None
    if args.w_pop != 0.0 and pop_z is None:
        print("[hybrid_fusion] popularity weight requested but no train data found; forcing w_pop=0", flush=True)
        args.w_pop = 0.0

    val_gt  = build_eval(args.val,  args.threshold)
    test_gt = build_eval(args.test, args.threshold)

    def parse_grid(values):
        return [float(v) for v in str(values).split(",") if v != ""]

    if args.tune_weights:
        cf_grid  = parse_grid(args.grid_cf)
        cbf_grid = parse_grid(args.grid_cbf)
        pop_grid = parse_grid(args.grid_pop)
        if pop_z is None:
            pop_grid = [w for w in pop_grid if w == 0.0] or [0.0]
        metric_key = f"{args.tune_metric}@{args.k_top}"
        best_tuple = None
        best_metric = -float("inf")
        for w_cf in cf_grid:
            for w_cbf in cbf_grid:
                for w_pop in pop_grid:
                    if w_cf == 0 and w_cbf == 0 and w_pop == 0:
                        continue
                    _, m = fuse_one_split(
                        args.cf_val, args.cbf_val, val_gt, args.k_top,
                        w_cf, w_cbf, w_pop, pop_z,
                        args.limit_users_val, out_csv=None
                    )
                    if metric_key not in m:
                        raise KeyError(f"metric '{metric_key}' not in validation metrics: {list(m.keys())}")
                    score = m[metric_key]
                    if score > best_metric:
                        best_metric = score
                        best_tuple = (w_cf, w_cbf, w_pop, m)
        if best_tuple is None:
            raise RuntimeError("no valid weight combination found during tuning")
        args.w_cf, args.w_cbf, args.w_pop, best_metrics = best_tuple
        print(json.dumps({
            "tuning": {
                "metric": metric_key,
                "best_weights": {"w_cf": args.w_cf, "w_cbf": args.w_cbf, "w_pop": args.w_pop},
                "best_val_metrics": best_metrics
            }
        }, indent=2))

    val_recs,  val_metrics  = fuse_one_split(
        args.cf_val, args.cbf_val, val_gt, args.k_top,
        args.w_cf, args.w_cbf, args.w_pop, pop_z,
        args.limit_users_val, out_csv=outdir/"val_recs_hybrid.csv"
    )
    test_recs, test_metrics = fuse_one_split(
        args.cf_test, args.cbf_test, test_gt, args.k_top,
        args.w_cf, args.w_cbf, args.w_pop, pop_z,
        args.limit_users_test, out_csv=outdir/"test_recs_hybrid.csv"
    )

    results = {
        "model": "Hybrid late-fusion (z-norm)",
        "weights": {"w_cf": args.w_cf, "w_cbf": args.w_cbf, "w_pop": args.w_pop},
        "k_top": args.k_top,
        "threshold": args.threshold,
        "val":  val_metrics,
        "test": test_metrics,
        "files": {
            "val_recs":  str((outdir/"val_recs_hybrid.csv").resolve()),
            "test_recs": str((outdir/"test_recs_hybrid.csv").resolve())
        }
    }
    (outdir/"hybrid_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
