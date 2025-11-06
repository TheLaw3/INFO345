# Greger/cbf_tfidf_fast.py — CBF TF-IDF with candidate pool + precomputed user likes
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

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

def load_splits(train_path, val_path, test_path):
    train = pd.read_csv(train_path)
    val   = pd.read_csv(val_path)
    test  = pd.read_csv(test_path)
    cleaned = []
    for df in (train, val, test):
        df = df.dropna(subset=["user_id", "item_id"]).copy()
        df["user_id"] = df["user_id"].astype(str).str.strip()
        df["item_id"] = df["item_id"].astype(str).str.strip()
        df["rating"]  = pd.to_numeric(df["rating"], errors="coerce").clip(1,5)
        df = df.dropna(subset=["rating"])
        df = df.drop_duplicates(subset=["user_id","item_id"], keep="last")
        cleaned.append(df)
    train, val, test = cleaned
    return train, val, test

def pick(colnames, candidates):
    s = {c.lower(): c for c in colnames}
    for c in candidates:
        if c in colnames: return c
        if c.lower() in s: return s[c.lower()]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val",   required=True)
    ap.add_argument("--test",  required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--k_top", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=4.0)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--max_features", type=int, default=30000)
    ap.add_argument("--ngram_max", type=int, default=1)
    ap.add_argument("--stop_words", default="english")   # "none" to disable
    ap.add_argument("--cand_pool", type=int, default=20000)  # cap candidate items
    ap.add_argument("--limit_users_val",  type=int, default=1000)
    ap.add_argument("--limit_users_test", type=int, default=1000)
    ap.add_argument("--outdir", default="out/cbf_fast")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    train, val, test = load_splits(args.train, args.val, args.test)

    items = pd.read_csv(args.items)
    id_col = "item_id" if "item_id" in items.columns else "Id"
    items["item_id"] = items[id_col].astype(str).str.strip()
    tcol = args.text_col
    if tcol not in items.columns:
        title = pick(items.columns, ["title","Title","name"])
        cats  = pick(items.columns, ["categories","category","genres","labels"])
        title_series = items.get(title, pd.Series("", index=items.index)).astype(str)
        cats_series  = items.get(cats,  pd.Series("", index=items.index)).astype(str)
        items[tcol] = (title_series + " " + cats_series).str.strip()
    items[tcol] = items[tcol].fillna("").astype(str).str.strip()
    empty = items[tcol].str.len() == 0
    if empty.any():
        items.loc[empty, tcol] = items.loc[empty, "item_id"]
    items = (items.assign(_text_len=items[tcol].str.len())
                  .sort_values(["item_id", "_text_len"], ascending=[True, False])
                  .drop_duplicates("item_id")
                  .drop(columns="_text_len"))
    item_ids = items["item_id"].tolist()
    idx_by_item = {iid:i for i, iid in enumerate(item_ids)}

    vec = TfidfVectorizer(min_df=args.min_df, max_features=args.max_features,
                          ngram_range=(1, args.ngram_max), stop_words=(None if args.stop_words=="none" else args.stop_words))
    X = vec.fit_transform(items[tcol].values)  # CSR [n_items, V]
    print(f"TFIDF: items={X.shape[0]}, terms={X.shape[1]}", flush=True)

    seen = {u: set(g["item_id"].astype(str).str.strip()) for u, g in train.groupby("user_id")}

    pop = train.groupby("item_id").size().sort_values(ascending=False)
    cand_head = set(pop.index.astype(str)[:args.cand_pool]).intersection(idx_by_item.keys())
    if not cand_head:
        cand_head = set(idx_by_item.keys())
    print(f"cand_pool={len(cand_head)}", flush=True)

    liked = train[(train["rating"] >= args.threshold) & (train["item_id"].isin(idx_by_item.keys()))].copy()
    liked["iid_idx"] = liked["item_id"].map(idx_by_item).astype("Int64")
    liked = liked.dropna(subset=["iid_idx"])
    liked["w"] = (liked["rating"] - args.threshold).clip(lower=0).astype(np.float32)
    grp = liked.groupby("user_id")
    liked_idx_by_u = {u: g["iid_idx"].astype(int).to_numpy() for u, g in grp}
    weights_by_u   = {u: g["w"].to_numpy()            for u, g in grp}

    def recommend_for_split(eval_df, split_name):
        users = eval_df["user_id"].astype(str).unique().tolist()
        users = [u for u in users if u in seen and u in liked_idx_by_u]
        if split_name == "val"  and args.limit_users_val:  users = users[:args.limit_users_val]
        if split_name == "test" and args.limit_users_test: users = users[:args.limit_users_test]
        print(f"{split_name}: users={len(users)}", flush=True)

        rows = []
        for i, u in enumerate(users, 1):
            li = liked_idx_by_u[u]
            w  = weights_by_u[u]
            if li.size == 0: 
                continue

            prof_vec = X[li].T.dot(w)                         # V-length ndarray
            prof = csr_matrix(prof_vec.reshape(1, -1))
            prof = normalize(prof, norm="l2", copy=False)

            cand_iids = sorted(cand_head - seen[u])
            if not cand_iids: 
                continue
            cand_idx = [idx_by_item[iid] for iid in cand_iids]
            Xcand = X[cand_idx]                               # Nc x V CSR
            scores = Xcand.dot(prof.T).toarray().ravel()      # Nc floats

            topk = min(args.k_top, scores.size)
            if topk <= 0: 
                continue
            top_idx = np.argpartition(-scores, topk-1)[:topk]
            top_sorted = top_idx[np.argsort(-scores[top_idx])]
            for r, j in enumerate(top_sorted, 1):
                rows.append({"user_id": u, "item_id": cand_iids[j],
                             "rank": r, "score": float(scores[j]), "source":"cbf_tfidf"})
            if i % 100 == 0:
                print(f"{split_name}: users_processed={i}, rows={len(rows)}", flush=True)
        return pd.DataFrame(rows)

    recs_val  = recommend_for_split(val,  "val")
    recs_test = recommend_for_split(test, "test")
    recs_val.to_csv(outdir/"val_recs_cbf.csv", index=False)
    recs_test.to_csv(outdir/"test_recs_cbf.csv", index=False)

    vr = val[val["rating"]  >= args.threshold][["user_id","item_id","rating"]]
    tr = test[test["rating"] >= args.threshold][["user_id","item_id","rating"]]
    res_val  = eval_topk(recs_val,  vr, args.k_top)
    res_test = eval_topk(recs_test, tr, args.k_top)

    metrics = {
        "model": "CBF TF-IDF (fast)",
        "params": {"k_top": args.k_top, "threshold": args.threshold,
                   "min_df": args.min_df, "max_features": args.max_features,
                   "ngram_max": args.ngram_max, "stop_words": args.stop_words,
                   "cand_pool": args.cand_pool,
                   "limit_users_val": args.limit_users_val,
                   "limit_users_test": args.limit_users_test},
        "topk": {"val": res_val, "test": res_test},
        "files": {
            "val_recs": str((outdir/"val_recs_cbf.csv").resolve()),
            "test_recs": str((outdir/"test_recs_cbf.csv").resolve()),
        }
    }
    (outdir/"cbf_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)

if __name__ == "__main__":
    main()
