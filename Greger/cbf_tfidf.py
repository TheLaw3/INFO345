# cbf_tfidf.py — Content-Based Filtering with TF-IDF (scikit-learn)
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ----- metrics -----
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

# ----- helpers -----
def load_splits(train_path, val_path, test_path):
    train = pd.read_csv(train_path)
    val   = pd.read_csv(val_path)
    test  = pd.read_csv(test_path)
    for df in (train, val, test):
        df["user_id"] = df["user_id"].astype(str)
        df["item_id"] = df["item_id"].astype(str)
        df["rating"]  = pd.to_numeric(df["rating"], errors="coerce").clip(1,5)
    return train, val, test

def build_seen(train_df):
    return {u: set(g["item_id"]) for u, g in train_df.groupby("user_id")}

def pick(colnames, candidates):
    s = {c.lower(): c for c in colnames}
    for c in candidates:
        if c in colnames: return c
        if c.lower() in s: return s[c.lower()]
    return None

# ----- CBF -----
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
    ap.add_argument("--max_features", type=int, default=100000)
    ap.add_argument("--ngram_max", type=int, default=2)
    ap.add_argument("--stop_words", default="english")   # use "none" to disable
    ap.add_argument("--limit_users_val",  type=int, default=None)
    ap.add_argument("--limit_users_test", type=int, default=None)
    ap.add_argument("--outdir", default="out/cbf")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    train, val, test = load_splits(args.train, args.val, args.test)

    # items + text
    items = pd.read_csv(args.items)
    items["item_id"] = items["item_id"].astype(str) if "item_id" in items.columns else items["Id"].astype(str)
    tcol = args.text_col
    if tcol not in items.columns:
        title = pick(items.columns, ["title","Title","name"])
        cats  = pick(items.columns, ["categories","category","genres","labels"])
        items[tcol] = items.get(title, pd.Series([""]*len(items))).astype(str) + " " + \
                      items.get(cats,  pd.Series([""]*len(items))).astype(str)
    items[tcol] = items[tcol].fillna("").astype(str)
    items = items.drop_duplicates("item_id")
    items = items[items[tcol].str.len() > 0]
    item_ids = items["item_id"].tolist()
    idx_by_item = {iid:i for i, iid in enumerate(item_ids)}

    # TF-IDF
    stop = None if args.stop_words.lower() == "none" else args.stop_words
    vec = TfidfVectorizer(min_df=args.min_df, max_features=args.max_features,
                          ngram_range=(1, args.ngram_max), stop_words=stop)
    X = vec.fit_transform(items[tcol].values)  # CSR [n_items, V]

    seen = build_seen(train)
    catalog_with_text = set(item_ids)

    def recommend_for_users(eval_df, split_name):
        users = eval_df["user_id"].unique().tolist()
        if split_name == "val" and args.limit_users_val:
            users = users[:args.limit_users_val]
        if split_name == "test" and args.limit_users_test:
            users = users[:args.limit_users_test]

        rows = []
        for u in users:
            if u not in seen: 
                continue
            utrain = train[(train["user_id"] == u) & (train["rating"] >= args.threshold)]
            utrain = utrain[utrain["item_id"].isin(catalog_with_text)]
            if utrain.empty:
                continue

            liked_idx = [idx_by_item[iid] for iid in utrain["item_id"] if iid in idx_by_item]
            if not liked_idx:
                continue

            w = np.maximum(utrain["rating"].to_numpy() - args.threshold, 0.0)

            subset = X[liked_idx]                         # L x V CSR
            subset = subset.multiply(w[:, None])          # weight rows
            prof_sum = subset.sum(axis=0)                 # 1 x V (matrix-like)
            prof_vec = prof_sum.toarray().ravel() if issparse(prof_sum) else np.asarray(prof_sum).ravel()
            prof = csr_matrix(prof_vec)                   # 1 x V CSR
            prof = normalize(prof, norm="l2", copy=False)

            cand_iids = list(catalog_with_text - seen[u])
            if not cand_iids:
                continue
            cand_idx = [idx_by_item[iid] for iid in cand_iids if iid in idx_by_item]
            if not cand_idx:
                continue

            Xcand = X[cand_idx]                           # Nc x V CSR
            scores_mat = Xcand @ prof.T                   # Nc x 1 sparse
            scores = scores_mat.toarray().ravel() if issparse(scores_mat) else np.asarray(scores_mat).ravel()

            topk = min(args.k_top, scores.size)
            if topk <= 0:
                continue
            top_idx = np.argpartition(-scores, topk-1)[:topk]
            top_sorted = top_idx[np.argsort(-scores[top_idx])]
            for r, local_i in enumerate(top_sorted, 1):
                rows.append({"user_id": u, "item_id": cand_iids[local_i],
                             "rank": r, "score": float(scores[local_i]),
                             "source": "cbf_tfidf"})
        return pd.DataFrame(rows)

    recs_val  = recommend_for_users(val,  "val")
    recs_test = recommend_for_users(test, "test")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    recs_val.to_csv(outdir/"val_recs_cbf.csv", index=False)
    recs_test.to_csv(outdir/"test_recs_cbf.csv", index=False)

    vr = val[val["rating"]  >= args.threshold][["user_id","item_id","rating"]]
    tr = test[test["rating"] >= args.threshold][["user_id","item_id","rating"]]
    res_val  = eval_topk(recs_val,  vr, args.k_top)
    res_test = eval_topk(recs_test, tr, args.k_top)

    metrics = {
        "model": "CBF TF-IDF",
        "params": {"k_top": args.k_top, "threshold": args.threshold,
                   "min_df": args.min_df, "max_features": args.max_features,
                   "ngram_max": args.ngram_max, "stop_words": args.stop_words},
        "topk": {"val": res_val, "test": res_test},
        "files": {
            "val_recs": str((outdir/"val_recs_cbf.csv").resolve()),
            "test_recs": str((outdir/"test_recs_cbf.csv").resolve()),
        }
    }
    (outdir/"cbf_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
