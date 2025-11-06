# cf_sklearn.py
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# ------- metrics -------
def ndcg_at_k(rec_items, rel_set, k):
    if k == 0: return 0.0
    dcg = 0.0
    for rank, iid in enumerate(rec_items[:k], start=1):
        if iid in rel_set:
            dcg += 1.0 / math.log2(rank + 1)
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
        f"recall@{k}": float(np.mean(recs)) if recs else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"hit_rate@{k}": float(np.mean(hits)) if hits else 0.0,
    }

# ------- CF model (item-kNN with cosine) -------
class ItemKNN:
    def __init__(self, n_neighbors=200, metric="cosine"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = NearestNeighbors(n_neighbors=n_neighbors+1, metric=metric, algorithm="brute")
        self.item_index = None
        self.user_index = None
        self.R = None  # CSR users x items
        self.neigh_ind = None
        self.neigh_sim = None
        self._inv_item_index = None
        self._seen = None

    def fit(self, train_df):
        # map ids
        users = train_df["user_id"].astype(str).unique()
        items = train_df["item_id"].astype(str).unique()
        self.user_index = {u:i for i,u in enumerate(users)}
        self.item_index = {it:j for j,it in enumerate(items)}
        self._inv_item_index = {j: it for it, j in self.item_index.items()}

        # build matrix
        ui = train_df["user_id"].map(self.user_index).values
        ii = train_df["item_id"].map(self.item_index).values
        rr = train_df["rating"].astype(float).values
        self.R = csr_matrix((rr, (ui, ii)), shape=(len(users), len(items)))

        # item vectors = columns -> shape (n_items, n_users)
        X = self.R.T  # CSR items x users
        self.model.fit(X)

        # precompute neighbors
        distances, indices = self.model.kneighbors(X, return_distance=True)
        # cosine distance d => similarity s = 1 - d
        self.neigh_ind = indices
        self.neigh_sim = 1.0 - distances

        # cache items seen by each user (string ids)
        self._seen = {str(u): set(g["item_id"].astype(str).str.strip())
                      for u, g in train_df.groupby("user_id")}

    def recommend_for_users(self, eval_users, K=10, max_neigh=None):
        if max_neigh is None or max_neigh > self.n_neighbors:
            max_neigh = self.n_neighbors
        rows = []
        if self._seen is None:
            self._seen = {}

        for u in eval_users:
            u = str(u)
            if u not in self.user_index:
                continue
            uidx = self.user_index[u]
            # user rated items and ratings
            start = self.R.indptr[uidx]
            end   = self.R.indptr[uidx+1]
            rated_items = self.R.indices[start:end]
            rated_vals  = self.R.data[start:end]
            if rated_items.size == 0:
                continue

            scores = {}
            seen_u = self._seen.get(u, set())
            for i_idx, r in zip(rated_items, rated_vals):
                neighs = self.neigh_ind[i_idx, 1:max_neigh+1]   # skip self at [0]
                sims   = self.neigh_sim[i_idx, 1:max_neigh+1]
                for j_idx, s in zip(neighs, sims):
                    iid = self._inv_item_index.get(j_idx)
                    if iid is None:
                        continue
                    if iid in seen_u:
                        continue
                    scores[iid] = scores.get(iid, 0.0) + float(s) * float(r)

            if not scores:
                continue
            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:K]
            rows += [{"user_id": u, "item_id": iid, "rank": r+1, "score": float(sc), "source":"sklearn_knn"}
                     for r, (iid, sc) in enumerate(top)]
        return pd.DataFrame(rows)

def load_splits(train_path, val_path, test_path):
    train = pd.read_csv(train_path)
    val   = pd.read_csv(val_path)
    test  = pd.read_csv(test_path)
    for df in (train, val, test):
        df["user_id"] = df["user_id"].astype(str).str.strip()
        df["item_id"] = df["item_id"].astype(str).str.strip()
        df["rating"]  = pd.to_numeric(df["rating"], errors="coerce").clip(1,5)
    return train, val, test

def log_relevance_coverage(split_df, split_name, item_index, threshold):
    rel = split_df[split_df["rating"] >= threshold][["user_id", "item_id"]].copy()
    if rel.empty:
        print(f"{split_name}: no interactions above threshold={threshold}", flush=True)
        return

    rel["item_id"] = rel["item_id"].astype(str).str.strip()
    rel["covered"] = rel["item_id"].isin(item_index)

    users_total = rel["user_id"].nunique()
    users_with_covered = rel.groupby("user_id")["covered"].any().sum()
    items_total = rel["item_id"].nunique()
    items_covered = rel.loc[rel["covered"], "item_id"].nunique()

    print(
        f"{split_name}: relevant_items={items_total}, covered_items={items_covered}, "
        f"users_with_covered={users_with_covered}/{users_total}",
        flush=True,
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val",   required=True)
    ap.add_argument("--test",  required=True)
    ap.add_argument("--k_top", type=int, default=10)
    ap.add_argument("--neighbors", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=4.0)
    ap.add_argument("--outdir", default="out/cf_sklearn")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    train, val, test = load_splits(args.train, args.val, args.test)

    # fit item-kNN
    model = ItemKNN(n_neighbors=args.neighbors, metric="cosine")
    model.fit(train)

    item_ids = set(model.item_index.keys())
    log_relevance_coverage(val, "val", item_ids, args.threshold)
    log_relevance_coverage(test, "test", item_ids, args.threshold)

    # build relevance sets
    val_rel  = val[val["rating"]  >= args.threshold][["user_id","item_id","rating"]]
    test_rel = test[test["rating"] >= args.threshold][["user_id","item_id","rating"]]

    # recommend for val/test users
    val_users  = val["user_id"].unique().tolist()
    test_users = test["user_id"].unique().tolist()
    recs_val  = model.recommend_for_users(val_users,  K=args.k_top)
    recs_test = model.recommend_for_users(test_users, K=args.k_top)
    recs_val.to_csv(outdir/"val_recs_knn_sklearn.csv", index=False)
    recs_test.to_csv(outdir/"test_recs_knn_sklearn.csv", index=False)

    # evaluate Top-K
    res_val  = eval_topk(recs_val,  val_rel,  args.k_top)
    res_test = eval_topk(recs_test, test_rel, args.k_top)

    # rating prediction baselines for reference (not primary for KNN ranking)
    # you can compute a weighted score RMSE if needed, but Top-K is main.

    metrics = {
        "model": "item-kNN sklearn cosine",
        "params": {"neighbors": args.neighbors, "k_top": args.k_top},
        "topk": {"val": res_val, "test": res_test},
        "files": {
            "val_recs": str((outdir/"val_recs_knn_sklearn.csv").resolve()),
            "test_recs": str((outdir/"test_recs_knn_sklearn.csv").resolve()),
        }
    }
    (outdir/"cf_sklearn_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
