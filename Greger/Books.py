import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

#  paths 
meta_path = Path("books_data.csv")      
ratings_path = Path("Books_rating.csv")  

#  load 
meta = pd.read_csv(meta_path, low_memory=False)
ratings = pd.read_csv(ratings_path, low_memory=False)

print("META shape:", meta.shape)
print("RATINGS shape:", ratings.shape)
print("META cols:", list(meta.columns)[:30])
print("RATINGS cols:", list(ratings.columns)[:30])

#  helpers 
def pick(colnames, candidates):
    for c in candidates:
        if c in colnames: return c
        for cn in colnames:
            if cn.lower()==c.lower(): return cn
    return None

rcols = set(ratings.columns)
mcols = set(meta.columns)

user_col   = pick(rcols, ["user_id","User_id","reviewerID","profileName","user"])
item_col_r = pick(rcols, ["asin","ASIN","book_id","item_id","product_id","Id","id"])
rate_col   = pick(rcols, ["rating","Rating","Score","overall","rate","stars"])

item_col_m = pick(mcols, ["asin","ASIN","book_id","item_id","product_id","Id","id"])
title_col  = pick(mcols, ["title","Title","name"])

if not all([user_col, item_col_r, rate_col]):
    raise ValueError(f"Missing key columns. Found -> user:{user_col}, item:{item_col_r}, rating:{rate_col}")

if not item_col_m:
    raise ValueError("Missing item id in meta to join with ratings.")
if not title_col:
    print("Warning: no title column found in meta; continuing without titles.")

#  basic cleaning 
df = ratings[[user_col, item_col_r, rate_col]].copy()
df.columns = ["user_id","item_id","rating"]

# cast rating to numeric
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

# drop unusable rows
before = len(df)
df = df.dropna(subset=["user_id","item_id","rating"])
df["user_id"] = df["user_id"].astype(str)
df["item_id"] = df["item_id"].astype(str)
df = df.drop_duplicates(subset=["user_id","item_id"])  # keep one opinion per pair
print(f"Dropped {before - len(df)} rows due to NA/dupes.")

#  join titles 
m = meta[[item_col_m] + ([title_col] if title_col else [])].copy()
m.columns = ["item_id"] + (["title"] if title_col else [])
m["item_id"] = m["item_id"].astype(str)
df = df.merge(m, on="item_id", how="left")

#  dataset overview 
n_users = df["user_id"].nunique()
n_items = df["item_id"].nunique()
n_inter = len(df)
density = n_inter / (n_users * n_items)
print(f"#users={n_users:,}  #items={n_items:,}  #interactions={n_inter:,}  density={density:.6f}")

print("Rating stats:")
print(df["rating"].describe())

miss_title = df["title"].isna().mean() if "title" in df else np.nan
print(f"Missing titles: {miss_title:.2%}" if "title" in df else "No titles available.")

#  popularity and activity 
user_activity = df.groupby("user_id").size().rename("n_ratings")
item_pop = df.groupby(["item_id","title"]).size().rename("n_ratings").reset_index() if "title" in df else \
           df.groupby("item_id").size().rename("n_ratings").reset_index()

print("\nTop 10 most-rated items:")
print(item_pop.sort_values("n_ratings", ascending=False).head(10))

#  plots 
plt.figure()
df["rating"].plot.hist(bins=20)
plt.title("Rating distribution")
plt.xlabel("rating")
plt.ylabel("count")
plt.tight_layout()
plt.show()

plt.figure()
user_activity.plot.hist(bins=50)
plt.title("User activity distribution (#ratings per user)")
plt.xlabel("#ratings")
plt.ylabel("users")
plt.tight_layout()
plt.show()

plt.figure()
item_pop["n_ratings"].plot.hist(bins=50)
plt.title("Item popularity distribution (#ratings per item)")
plt.xlabel("#ratings")
plt.ylabel("items")
plt.tight_layout()
plt.show()

#  cold-start flags 
cold_users = (user_activity==1).mean()
cold_items = (item_pop["n_ratings"]==1).mean()
print(f"Cold-start users (only 1 rating): {cold_users:.2%}")
print(f"Cold-start items (only 1 rating): {cold_items:.2%}")

#  optional filters for modeling preview 
MIN_U, MIN_I = 3, 3  
mask_u = df["user_id"].isin(user_activity[user_activity>=MIN_U].index)
mask_i = df["item_id"].isin(item_pop[item_pop["n_ratings"]>=MIN_I]["item_id"])
df_filt = df[mask_u & mask_i].copy()

print(f"After filtering (u>={MIN_U}, i>={MIN_I}): {len(df_filt):,} rows,"
      f" users={df_filt['user_id'].nunique():,}, items={df_filt['item_id'].nunique():,}")

# save cleaned
df_filt.to_csv("ratings_clean.csv", index=False)
print("Wrote ratings_clean.csv")
