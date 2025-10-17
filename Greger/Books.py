import pandas as pd
from pathlib import Path

meta_path = Path("books_data.csv")       # change if needed
ratings_path = Path("Books_rating.csv")  # change if needed

meta = pd.read_csv(meta_path, low_memory=False)
ratings = pd.read_csv(ratings_path, low_memory=False)

print("META shape:", meta.shape)
print("META columns:", list(meta.columns))
print("RATINGS shape:", ratings.shape)
print("RATINGS columns:", list(ratings.columns))

