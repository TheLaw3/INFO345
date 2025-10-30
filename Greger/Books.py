import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Paths
meta_path = Path("/Users/sondrebjerck/Documents/INFO354Dataset/books_data.csv")
ratings_path = Path("/Users/sondrebjerck/Documents/INFO354Dataset/ratings_clean.csv")

if not meta_path.exists():
    raise FileNotFoundError(f"Missing file: {meta_path}")
if not ratings_path.exists():
    raise FileNotFoundError(f"Missing file: {ratings_path}")

# Load
meta = pd.read_csv(meta_path, low_memory=False)
ratings = pd.read_csv(ratings_path, low_memory=False)

# Helper to find columns by common aliases
def pick(colnames, candidates):
    s = set(colnames)
    for c in candidates:
        if c in s:
            return c
        for cn in colnames:
            if cn.lower() == c.lower():
                return cn
    return None

# Identify columns
rcols = list(ratings.columns)
mcols = list(meta.columns)

user_col    = pick(rcols, ["user_id","User_id","reviewerID","profileName","user"])
item_col_r  = pick(rcols, ["asin","ASIN","book_id","item_id","product_id","Id","id"])
rate_col    = pick(rcols, ["rating","Rating","Score","overall","rate","stars","review/score"])
title_col_r = pick(rcols, ["title","Title","name","book_title"])

item_col_m  = pick(mcols, ["asin","ASIN","book_id","item_id","product_id","Id","id"])
title_col_m = pick(mcols, ["title","Title","name"])

if not all([user_col, item_col_r, rate_col]):
    raise ValueError(f"Missing key columns. Found -> user:{user_col}, item:{item_col_r}, rating:{rate_col}")

# Build base ratings frame
keep = [user_col, item_col_r, rate_col] + ([title_col_r] if title_col_r else [])
df = ratings[keep].copy()
df.columns = ["user_id","item_id","rating"] + (["title_r"] if title_col_r else [])

# Clean types and duplicates
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna(subset=["user_id","item_id","rating"])
df["user_id"] = df["user_id"].astype(str)
df["item_id"] = df["item_id"].astype(str)
df = df.drop_duplicates(subset=["user_id","item_id"])

# Attach a title for readability. Prefer ID join; fall back to Title normalize.
joined = False
if item_col_m and item_col_m == item_col_r:
    m = meta[[item_col_m] + ([title_col_m] if title_col_m else [])].copy()
    m.columns = ["item_id"] + (["title"] if title_col_m else [])
    m["item_id"] = m["item_id"].astype(str)
    df = df.merge(m, on="item_id", how="left")
    joined = True
elif title_col_r and title_col_m:
    m = meta[[title_col_m]].copy()
    m.columns = ["title_m"]
    df["title_norm"] = df["title_r"].astype(str).str.lower().str.strip()
    m["title_norm"] = m["title_m"].astype(str).str.lower().str.strip()
    df = df.merge(m[["title_norm","title_m"]], on="title_norm", how="left")
    df["title"] = df["title_m"].where(df["title_m"].notna(), df["title_r"])
    df = df.drop(columns=[c for c in ["title_r","title_m","title_norm"] if c in df])
    joined = True

if not joined and "title_r" in df:
    df["title"] = df["title_r"]
    df = df.drop(columns=["title_r"])

# Shapes and columns
print("META shape:", meta.shape)
print("RATINGS shape:", ratings.shape)
print("META cols:", mcols)
print("RATINGS cols:", rcols)

# Basic stats
n_users = df["user_id"].nunique()
n_items = df["item_id"].nunique()
n_inter = len(df)
density = n_inter / (n_users * n_items)
print(f"#users={n_users:,}  #items={n_items:,}  #interactions={n_inter:,}  density={density:.6f}")

print("Rating summary:")
print(df["rating"].describe())

if "title" in df:
    print(f"Missing titles: {df['title'].isna().mean():.2%}")
else:
    print("No titles available after join.")

# Distributions
print("\nRating shares:")
print(df["rating"].value_counts(normalize=True).sort_index().round(3))

user_activity = df.groupby("user_id").size()
item_pop = df.groupby("item_id").size()

print(f"\nUsers with 1 rating: {(user_activity.eq(1).mean()*100):.1f}%")
print(f"Items with 1 rating: {(item_pop.eq(1).mean()*100):.1f}%")

# Pareto-style concentration
pop_sorted = item_pop.sort_values(ascending=False).values
cut = max(1, int(0.20 * len(pop_sorted)))
share_top20 = pop_sorted[:cut].sum() / pop_sorted.sum()
print(f"Top 20% items generate {share_top20:.2%} of all ratings")

# Top items
print("\nTop 10 items by number of ratings:")
top_items = (df.groupby(["item_id","title"]).size().rename("n_ratings").reset_index()
             if "title" in df else
             item_pop.rename("n_ratings").reset_index())
print(top_items.sort_values("n_ratings", ascending=False).head(10))

# Plots (linear)
plt.figure()
df["rating"].plot.hist(bins=20)
plt.title("Rating distribution")
plt.xlabel("rating"); plt.ylabel("count")
plt.tight_layout(); plt.show()

plt.figure()
user_activity.plot.hist(bins=50)
plt.title("User activity")
plt.xlabel("ratings per user"); plt.ylabel("users")
plt.tight_layout(); plt.show()

plt.figure()
item_pop.plot.hist(bins=50)
plt.title("Item popularity")
plt.xlabel("ratings per item"); plt.ylabel("items")
plt.tight_layout(); plt.show()

# Plots (log-log to see the tails)
bins_u = np.logspace(0, np.log10(max(1, user_activity.max())), 50)
plt.figure()
plt.hist(user_activity, bins=bins_u)
plt.xscale("log"); plt.yscale("log")
plt.title("User activity (log-log)")
plt.xlabel("#ratings per user"); plt.ylabel("users")
plt.tight_layout(); plt.show()

bins_i = np.logspace(0, np.log10(max(1, item_pop.max())), 50)
plt.figure()
plt.hist(item_pop, bins=bins_i)
plt.xscale("log"); plt.yscale("log")
plt.title("Item popularity (log-log)")
plt.xlabel("#ratings per item"); plt.ylabel("items")
plt.tight_layout(); plt.show()

# Filter for modeling and save
MIN_U, MIN_I = 5, 5
mask_u = df["user_id"].isin(user_activity[user_activity>=MIN_U].index)
mask_i = df["item_id"].isin(item_pop[item_pop>=MIN_I].index)
df_filt = df[mask_u & mask_i].copy()

print(f"\nAfter filters u>={MIN_U}, i>={MIN_I}: "
      f"{len(df_filt):,} rows, {df_filt.user_id.nunique():,} users, {df_filt.item_id.nunique():,} items")

out_path = Path("ratings_clean.csv")
df_filt.to_csv(out_path, index=False)
print(f"Wrote {out_path.resolve()}")
