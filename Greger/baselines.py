# baselines.py  — fast baselines with caps + progress logs
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

def log(msg): print(msg, flush=True)

# ---- metrics ----
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

# ---- helpers ----
def build_seen(train_df):
    return {str(u): set(g["item_id"].astype(str)) for u, g in train_df.groupby("user_id")}

def topk_popularity_stream(train_df, users, seen, K, out_csv, scan_limit=None, progress_every=2000):
    pop = train_df.groupby("item_id").size().sort_values(ascending=False)
    pop_items = pop.index.astype(str).tolist()
    if scan_limit: pop_items = pop_items[:scan_limit]
    rows = []
    for idx, u in enumerate(users, 1):
        su = seen.get(u); 
        if su is None: continue
        picked = []
        for iid in pop_items:       # scan only the head; stop as soon as K found
            if iid not in su:
                picked.append(iid)
                if len(picked) == K: break
        for r, iid in enumerate(picked, 1):
            rows.append({"user_id": u, "item_id": iid, "rank": r, "score": float(pop.get(iid, 0)), "source": "popularity"})
        if idx % progress_every == 0: log(f"[pop] users processed: {idx}")
    out = pd.DataFrame(rows); out.to_csv(out_csv, index=False)
    return out, pop

def topk_random(users, all_items_set, seen, K, out_csv, seed=42, progress_every=2000):
    rng = np.random.default_rng(seed)
    rows = []
    for idx, u in enumerate(users, 1):
        su = seen.get(u)
        if su is None: continue
        cand = list(all_items_set - su)
        if not cand: continue
        take = min(K, len(cand))
        choice = list(rng.choice(cand, size=take, replace=False))
        for r, iid in enumerate(choice, 1):
            rows.append({"user_id": u, "item_id": iid, "rank": r, "score": 0.0, "source": "random"})
        if idx % progress_every == 0: log(f"[rand] users processed: {idx}")
    out = pd.DataFrame(rows); out.to_csv(out_csv, index=False)
    return out

def rating_baselines(train_df, eval_df):
    gmean = train_df["rating"].mean()
    umean = train_df.groupby("user_id")["rating"].mean()
    imean = train_df.groupby("item_id")["rating"].mean()
    def preds(df, mode):
        if mode == "global": return np.full(len(df), gmean, dtype=float)
        if mode == "user":   return df["user_id"].map(umean).fillna(gmean).astype(float).values
        if mode == "item":   return df["item_id"].map(imean).fillna(gmean).astype(float).values
    metrics = {}
    for split_name, df in eval_df.items():
        if df is None or len(df) == 0: metrics[split_name] = {}; continue
        y = df["rating"].astype(float).values
        out = {}
        for mode in ["global","user","item"]:
            yhat = preds(df, mode)
            out[f"{mode}_rmse"] = float(np.sqrt(np.mean((y - yhat) ** 2)))
            out[f"{mode}_mae"]  = float(np.mean(np.abs(y - yhat)))
        metrics[split_name] = out
    return metrics

# ---- main ----
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val",   required=True)
    ap.add_argument("--test",  required=True)
    ap.add_argument("--items", default="")
    ap.add_argument("--k",     default="10")                 # e.g. "5,10"
    ap.add_argument("--threshold", type=float, default=4.0)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pop_scan_limit", type=int, default=20000)   # cap scanned head of popularity
    ap.add_argument("--limit_users_val",  type=int, default=None)  # quick smoke
    ap.add_argument("--limit_users_test", type=int, default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    log("load splits")
    train = pd.read_csv(args.train); val = pd.read_csv(args.val); test = pd.read_csv(args.test)
    for df in (train, val, test):
        df["user_id"] = df["user_id"].astype(str)
        df["item_id"] = df["item_id"].astype(str)
        df["rating"]  = pd.to_numeric(df["rating"], errors="coerce").clip(1,5)

    # catalog
    if args.items and Path(args.items).exists():
        items_df = pd.read_csv(args.items)
        all_items = items_df["item_id"].astype(str).unique().tolist() if "item_id" in items_df.columns \
                    else pd.unique(pd.concat([train["item_id"], val["item_id"], test["item_id"]]).astype(str)).tolist()
    else:
        all_items = pd.unique(pd.concat([train["item_id"], val["item_id"], test["item_id"]]).astype(str)).tolist()
    all_items_set = set(all_items)

    seen = build_seen(train)
    val_users  = val["user_id"].astype(str).unique().tolist()
    test_users = test["user_id"].astype(str).unique().tolist()
    if args.limit_users_val:  val_users  = val_users[:args.limit_users_val]
    if args.limit_users_test: test_users = test_users[:args.limit_users_test]
    K_list = [int(x) for x in args.k.split(",")]
    Kmax = max(K_list)

    log(f"users(val,test)=({len(val_users)},{len(test_users)}), items={len(all_items)}, Kmax={Kmax}, scan_limit={args.pop_scan_limit}")

    # build recs
    log("popularity val");  pop_recs_val, pop_counts = topk_popularity_stream(train, val_users,  seen, Kmax, outdir/"val_recs_pop.csv",  scan_limit=args.pop_scan_limit)
    log("popularity test"); pop_recs_test, _         = topk_popularity_stream(train, test_users, seen, Kmax, outdir/"test_recs_pop.csv", scan_limit=args.pop_scan_limit)
    log("random val");      rand_recs_val  = topk_random(val_users,  all_items_set, seen, Kmax, outdir/"val_recs_rand.csv",  seed=args.seed)
    log("random test");     rand_recs_test = topk_random(test_users, all_items_set, seen, Kmax, outdir/"test_recs_rand.csv", seed=args.seed)

    # evaluation
    val_rel  = val[val["rating"]  >= args.threshold][["user_id","item_id","rating"]]
    test_rel = test[test["rating"] >= args.threshold][["user_id","item_id","rating"]]
    results = {"topk": {"val": {}, "test": {}}, "rating": {}}
    for K in K_list:
        results["topk"]["val"][f"K={K}"] = {"popularity": eval_topk(pop_recs_val, val_rel, K),
                                            "random":     eval_topk(rand_recs_val, val_rel, K)}
        results["topk"]["test"][f"K={K}"] = {"popularity": eval_topk(pop_recs_test, test_rel, K),
                                             "random":     eval_topk(rand_recs_test, test_rel, K)}

    # coverage/novelty for test Kmax
    pop_rank = pd.Series(range(1, len(pop_counts)+1), index=pop_counts.index.astype(str))
    recs_k = pop_recs_test[pop_recs_test["rank"] <= Kmax]
    cov = recs_k["item_id"].nunique() / max(1, len(all_items))
    pr = recs_k["item_id"].map(pop_rank).dropna()
    novelty = float(np.mean(1.0 - (pr - 1) / max(1, len(pop_rank)-1))) if len(pr) else 0.0
    results["topk"]["test"][f"K={Kmax}"]["popularity"]["catalog_coverage"] = cov
    results["topk"]["test"][f"K={Kmax}"]["popularity"]["novelty_percentile"] = novelty

    # rating baselines
    results["rating"] = rating_baselines(train, {"val": val, "test": test})

    # save
    (outdir/"baseline_metrics.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame([
        {"model":"popularity",
         f"precision@{Kmax}":results["topk"]["test"][f"K={Kmax}"]["popularity"][f"precision@{Kmax}"],
         f"recall@{Kmax}":   results["topk"]["test"][f"K={Kmax}"]["popularity"][f"recall@{Kmax}"],
         f"ndcg@{Kmax}":     results["topk"]["test"][f"K={Kmax}"]["popularity"][f"ndcg@{Kmax}"],
         f"hit_rate@{Kmax}": results["topk"]["test"][f"K={Kmax}"]["popularity"][f"hit_rate@{Kmax}"],
         "coverage": cov, "novelty_percentile": novelty},
        {"model":"random",
         f"precision@{Kmax}":results["topk"]["test"][f"K={Kmax}"]["random"][f"precision@{Kmax}"],
         f"recall@{Kmax}":   results["topk"]["test"][f"K={Kmax}"]["random"][f"recall@{Kmax}"],
         f"ndcg@{Kmax}":     results["topk"]["test"][f"K={Kmax}"]["random"][f"ndcg@{Kmax}"],
         f"hit_rate@{Kmax}": results["topk"]["test"][f"K={Kmax}"]["random"][f"hit_rate@{Kmax}"],
         "coverage": np.nan, "novelty_percentile": np.nan}
    ]).to_csv(outdir/"baseline_test_summary.csv", index=False)

    log("done")
    log(str((outdir/"baseline_metrics.json").resolve()))
    log(str((outdir/"baseline_test_summary.csv").resolve()))
