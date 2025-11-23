# Reproducibility
This section specifies the code, data, environment, and commands
used in the experiments. Link: https://github.com/TheLaw3/INFO345
Code repository
The main entry points are:
-src/prepare_data.py: data cleaning and construction of
CF/CBF-ready tables.
-src/split.py: creation of train/validation/test splits.
-src/baselines.py: random and popularity baselines.
-src/cbf_tfidf.py: content-based TF–IDF recommender.
-src/cf_sklearn.py: item-kNN collaborative filtering.
-src/hybrid_fusion.py: weighted late-fusion hybrid (CF +
CBF + popularity).
Dataset and external download
We use the Amazon Books ReviewsAmazon Book Reviews dataset
from Kaggle:
https://www.kaggle.com/datasets/mohamedbakhet/amazon-
books-reviews/data
The raw files are not included in the repository because of dataset
terms and file size. To reproduce the project:
(1) Create a Kaggle account and accept the dataset terms.
(2) Download the two CSV files from the “Data” tab:
- the ratings / reviews file (user–book interactions),
- the books metadata file (titles, descriptions, authors, cate-
gories, etc.).
(3) Place these two files in the project root and rename them
to ratings_clean.csv (ratings file) and books_data.csv
(metadata file), or adapt the filenames in the –ratings /
–items arguments below.
Software environment
Experiments were run on a 64-bit laptop with macOS and Python 3.11
(Anaconda). The code depends on the following Python libraries:
Python 3.11.x pandas (data loading and tabular processing) numpy
(numerical operations and random number generation) scikit-learn
(TF–IDF vectorizer, item-kNN, cosine similarity) argparse, pathlib
(standard-library modules for CLI and paths)
A minimal environment can be created, for example, with:
conda create -n info345-py311 python=3.11
conda activate info345-py311
pip install pandas numpy scikit-learn
Random seeds
All random seeds are fixed to 42. The splitting script src/split.py
and the baseline script src/baselines.py both expose a –seed
argument (default 42), and this default is used in all reported runs.
Any internal use of NumPy/Python RNG also uses seed 42.
Step-by-step commands
From the repository root, the full pipeline (data preparation, base-
lines, CBF, CF, hybrid) can be executed with:
1) Prepare CF/CBF-ready ratings and item tables
python src/prepare_data.py \
  --ratings ratings_clean.csv \
  --items   books_data.csv \
  --outdir  data \
  --min_user 5 \
  --min_item 5

2) Train/validation/test split (hold-out, seed=42)
python src/split.py \
  --ratings data/trainable_ratings.csv \
  --outdir  data \
  --seed 42

3) Baseline recommenders (popularity + random)
python src/baselines.py \
  --train data/train.csv \
  --val   data/val.csv \
  --test  data/test.csv \
  --items data/items.csv \
  --outdir out/baselines \
  --seed 42

4) Content-based TF–IDF recommender
python src/cbf_tfidf.py \
  --train data/train.csv \
  --val   data/val.csv \
  --test  data/test.csv \
  --items data/items.csv \
  --outdir out/cbf

5) Item-kNN collaborative filtering (sklearn)
python src/cf_sklearn.py \
  --train data/train.csv \
  --val   data/val.csv \
  --test  data/test.csv \
  --outdir out/cf_sklearn

6) Weighted late-fusion hybrid (CF + CBF + popularity)
python src/hybrid_fusion.py \
  --train   data/train.csv \
  --val     data/val.csv \
  --test    data/test.csv \
  --cf_val  out/cf_sklearn/val_recs_knn_sklearn.csv \
  --cf_test out/cf_sklearn/test_recs_knn_sklearn.csv \
  --cbf_val out/cbf/val_recs_cbf.csv \
  --cbf_test out/cbf/test_recs_cbf.csv \
  --k_top 10 \
  --threshold 4.0 \
  --w_cf 0.9 --w_cbf 0.4 --w_pop 0.0 \
  --outdir out/hybrid





# Ingest
"""Ingest raw ratings and items CSVs into a project data directory and emit a schema report.

This script reads the raw ratings and item metadata CSVs, makes byte‑exact
copies in a dedicated output directory (`data/raw` by default) and writes a
JSON report describing the datasets (file paths, row/column counts, column
names and simple column hints). 

It accepts three command‑line arguments:
`--ratings` (path to the user–item ratings CSV), 
`--items` (path to the
item metadata CSV) and 
`--outdir` (where the copied files and report are saved). 
By duplicating the raw inputs and recording their structure, the script ensures that downstream preprocessing, 
analysis and modelling can always be traced back to a fixed snapshot of the original data.

refrences 

Lectures 1 & 2 Introduction to Recommender Systems emphasise that raw
  recommender datasets are sparse and often messy; understanding dataset
  size, sparsity and column semantics is a prerequisite for building any
  model.  Copying the original ratings and item files into a stable
  directory and summarising their structure aligns with the course advice
  to document data characteristics before starting exploratory analysis.


Outputs 
  books_rating_cleaned.raw.csv  Unmodified copy of the ratings source.
  books_data.raw.csv            Unmodified copy of the items source.
  ingest_report.json            Summary of shapes, columns, and soft column hints.

Libraries 
pandas: Robust CSV I/O and dtype handling for large files (used here as a dependable reader/writer).
  Alternative: polars (faster on large data, different API; not necessary for a single pass copy and report).
pathlib: Cross-platform, explicit filesystem paths. Alternative: os.path (works, but less ergonomic).
argparse/json: Standard library CLIs and structured reports. Alternatives: click/typer 

"""

# prepare data
"""Prepare ratings and item metadata for CF/CBF pipelines and emit a preprocessing report.

This script ingests raw ratings and item metadata, cleans and standardises
column names, builds a consistent item catalog with enriched text fields
for content‑based filtering, filters users and items by minimum activity
thresholds for collaborative filtering, and writes three canonical files:

1. `trainable_ratings.csv` containing all cleaned interactions with
canonical columns (`user_id`, `item_id`, `rating`, `timestamp`);

2.`items.csv` containing the catalog of items with title, categories
and a concatenated `text` field for CBF models; and 

3. `ratings_cf_train.csv` containing the subset of interactions involving users and items that
meet the `min_user` and `min_item` thresholds for CF models.  

A JSON file summarising dataset sizes, sparsity, file paths and split
parameters is also written to assist reproducibility.

refrences 

Lectures 1 & 2 – Introduction.
  Introduce the concept of sparsity in recommender datasets and explain why
  it is important to count users, items and interactions before modelling.
  The script prints these counts and the density to help understand the
  dataset’s sparsity and head/tail distributions.
  
  abel Your Data (2025), “Data Versioning: Best Practices for ML
  Engineers.”
  The article notes that a straightforward data versioning strategy is
  full duplication: saving a complete copy of the dataset whenever it
  changes; each copy acts as a snapshot:contentReference. It also
  recommends capturing metadata (schema, column names) and linking data
  versions with code and experiments, and automating versioning in
  pipelines to produce traceable snapshots:contentReference. Our
  ingestion script duplicates the raw CSVs, and this module documents the
  schema and shapes of the processed datasets in a report for
  traceability.
  URL: https://labelyourdata.com/articles/machine-learning/data-versioning 
  
  

This script:
  1) Loads raw ratings and item metadata CSVs.
  2) Maps heterogeneous column names onto canonical fields: user_id, item_id, rating, and optional title.
  3) Cleans ratings (numeric coercion, clamp to [1, 5], drop NA, strip IDs, deduplicate user-item pairs).
  4) Builds an item catalog and enriches it with optional title/categories from metadata via normalized Title match.
  5) Constructs a 'text' field per item for content-based filtering (CBF).
  6) Produces a CF-ready ratings subset using user/item activity thresholds.
  7) Saves artifacts and a JSON summary report.

Inputs:
  --ratings  Raw ratings CSV path (default: data/raw/books_rating_cleaned.raw.csv)
  --items    Raw items CSV path   (default: data/raw/books_data.raw.csv)
  --outdir   Output directory     (default: data)
  --min_user Minimum interactions per user to keep in CF subset (default: 5)
  --min_item Minimum interactions per item to keep in CF subset (default: 5)

Outputs:
  trainable_ratings.csv   Cleaned ratings with canonical columns.
  items.csv               Item catalog with text field and optional Title/Categories.
  ratings_cf_train.csv    CF-ready subset filtered by min_user/min_item.
  preprocess_report.json  Row/user/item counts and file paths.

Libraries:
  pandas: ETL, CSV I/O, grouping, joins. Chosen for readability and course alignment.
    Alternative: polars (faster; different API; not essential for this pipeline scale).
  pathlib: Cross-platform path handling. Alternative: os.path (less ergonomic).
  argparse/json: Standard library for CLIs and structured reports. Alternatives: click/typer (extra deps).

"""




# split.py
"""Split cleaned ratings into train/val/test by user using random or temporal strategy.

This module reads a cleaned ratings file (with columns `user_id`, `item_id`,
`rating` and an timestamp) and partitions each user’s interactions
into three disjoint sets. Two strategies are supported:

Random per‑user split: For each user, one interaction is held out for the
  test set, and—if the user has at least three interactions—a second is held
  out for validation. A coverage-aware heuristic (`pick_idx_prefer_supported`)
  prefers to hold out interactions on items with higher global support so that
  rare items remain in the training set. Remaining interactions form the
  training set.

Temporal per‑user split: Interactions for each user are sorted by
  timestamp; the most recent interaction is placed in the test set, the
  second‑most recent (if present) becomes validation, and the rest form the
  training set.

After splitting, the script filters out users with fewer than
`min_train_inter` training interactions. It writes three CSVs—`train.csv`,
`val.csv`, `test.csv`—and a `split_stats.json


Strategies:
  - random: per user, select one test row and one validation row,
            preferring items that keep catalog coverage intact using a heuristic.
  - temporal: per user, last interaction → test, second-to-last → val,
              remaining → train.

Outputs:
  - train.csv
  - val.csv
  - test.csv
  - split_stats.json

"""

# eda
"""src/eda.py — quick exploratory analysis for the standardized data.

  This script performs a quick, deterministic EDA on the training data used
for our recommender models. It reads a ratings CSV (user_id, item_id,
rating) and, an items CSV containing item metadata, cleans and
merges the data, computes summary statistics, and generates
histograms to visualize rating distributions, user activity and item
popularity. The results are written to a JSON file and, if requested,
PNG plots are saved. The goal is to understand dataset size, sparsity,
distributional properties and potential cold‑start issues before building
recommendation models.

refrences 

Lectures 1 & 2 – Introduction to Recommender Systems.  
  The introductory slides emphasise that recommendation datasets are
  typically very sparse (only a tiny fraction of the user–item matrix is
  filled) and exhibit a long‑tail distribution of user activity and item
  popularity.  Calculating `n_users`, `n_items`, `n_interactions` and
  density (`|R| / (|U|·|I|)`) helps us gauge sparsity and anticipate
  challenges for CF/CBF models.  Plotting histograms of user activity and
  item popularity (both linear and log–log) reveals the head–tail
  imbalance discussed in the lectures.  The percentage of users/items
  with a single rating and the top‑20% popularity share quantify coldness
  and head concentration.
  
  analytics Vidhya, “Step‑by‑step Exploratory Data Analysis using Python.” 
   This article defines EDA as the process of performing initial
   investigations on data to discover patterns and check assumptions with
   summary statistics and graphical representations; EDA can be leveraged
   to identify outliers, patterns, trends and clues for imputing missing
   values. It notes that statistics summaries
   (count, mean, std, etc.) help identify outliers and skewness.
   We follow this guidance by computing `rating_summary`, `rating_shares`
   and percentages of users/items with a single interaction, and by using
   histograms to visualize distributions.
   URL: https://www.analyticsvidhya.com/blog/2022/07/step-by-step-exploratory-data-analysis-eda-using-python/

Shruti Udupa (2025), “First Try at Building a Recommendation System:
   Exploratory Data Analysis.” 
   In the EDA phase of a recommender project, Udupa emphasises three
   stages: 
   1. initial cleaning of datasets, treating missing values and
   identifying outliers; 
   2. in‑depth analysis and visualization of
   movie features, user behaviour patterns and rating distributions to
   uncover relationships;  
   3. engineering new predictive features or
   removing uninformative ones based on these insights.
   Our script mirrors this workflow: `load_ratings` cleans and coerces
   ratings, dropping invalid rows; `attach_titles` merges item metadata to
   enrich the data; the core analysis computes descriptive statistics and
   head–tail measures; optional plots visualize rating, user and item
   distributions; and the resulting statistics (e.g. density, cold‑user
   percentages, popularity share) inform subsequent modelling choices.
   URL: https://medium.com/pythoneers/first-try-at-building-an-end-to-end-recommendation-system-exploratory-data-analysis-c90cfd1b6ad6


Inputs/Outputs
  Inputs:
    - --ratings: CSV with columns {user_id, item_id, rating}.
    - --items:   items CSV with {item_id, Title/title, Categories/categories, text}.
  Outputs:
    - <outdir>/eda_stats.json          Dataset summary for reports and sanity checks.

What we measure and why
  n_users, n_items, n_interactions: dataset size and shape.
  density = |R| / (|U|·|I|): sparsity level, key for CF/CBF feasibility.
  rating_summary, rating_shares: label distribution; informs threshold=4.0 choice.
  users_with_single_rating_pct, items_with_single_rating_pct: coldness indicators.
  top20pct_popularity_share: head concentration; affects popularity baselines and novelty.

Libraries
  pandas: mature CSV/series ops. Alternative: polars (faster) but adds friction for course baselines.
  numpy: basic numerics and array ops; lightweight and ubiquitous.
  matplotlib: direct control for static PNGs; stable in headless mode with Agg.
    Alternatives: seaborn/plotnine (higher-level styling) not required for minimal, reproducible plots.
  pathlib/json/argparse: stdlib for paths, reports, and CLI; no extra deps.

"""


# baselines
"""Fast recommender baselines with progress logs and caps.

Refrences 

Lecture 7 – Offline Evaluation (PPTX)
  Slides 22–39: introduce rating-prediction error metrics such as MAE and
  RMSE (“Error Rate: MAE”, “Error Rate: RMSE”). We use these definitions
  when computing MAE/RMSE for our rating baselines (global/user/item mean).
  Slides 41 and 43: introduce Top-N recommendation tasks and evaluation
  for ranked recommendation lists.
  Slides 54, 56, 65 and 74: define and illustrate Precision@K, Recall@K,
  MAP@K and ranking-quality style metrics. Our evaluation code for the
  random and popularity baselines uses the same family of Top-N metrics
  (e.g. precision@10, recall@10, hit-rate@10, NDCG@10).

Ekstrand, M. D., Riedl, J. T., & Konstan, J. A. (2011).
  “Collaborative Filtering Recommender Systems.”
  Foundations and Trends in Human–Computer Interaction, 4(2), 81–173.
  DOI: https://doi.org/10.1561/1100000009
  Used for: formal definition of baseline predictors such as global mean,
  user mean and item mean (Section 2.1 “Baseline Predictors”), which
  motivates our rating baselines, and for background on evaluation of CF
  algorithms.
  
  Herlocker, J. L., Konstan, J. A., Terveen, L. G., & Riedl, J. T. (2004).
  “Evaluating Collaborative Filtering Recommender Systems.”
  ACM Transactions on Information Systems, 22(1), 5–53.
  DOI: https://doi.org/10.1145/963770.963772
  Used for: methodological guidance on offline evaluation design (choice
  of user tasks, datasets, and accuracy metrics) and for the practice of
  comparing new recommenders against simple non-personalized baselines
  such as popularity or random recommenders.

Implements two top-K recommenders:
Popularity (global most-interacted items, head scan capped).
Random (uniform from unseen items per user).

Also implements rating prediction baselines:
  Global/user/item mean with RMSE/MAE.

Top-K metrics:
  precision@k, recall@k, ndcg@k, hit_rate@k.
  Catalog coverage and novelty percentile for popularity at Kmax.

Why these libraries
  pandas: robust tabular IO and groupby ops used to compute popularity and clean splits.
    Alternatives: polars (faster on large data) was not chosen to minimize dependencies and
    keep parity with scikit-learn/pandas ecosystem.
  numpy: vectorized math and RNG for the random baseline; ubiquitous and stable.
    Alternatives: Python’s random module lacks vectorized sampling and is slower at scale.
  math: stable log2 for DCG; avoids bringing heavier deps.
  pathlib/json/argparse: stdlib for paths, structured reports, and CLI; no runtime cost.

"""



# Content-based filtering using TF-IDF item text with candidate cap.
src/cbf_tfidf.py — CBF TF-IDF (fixed types + candidate cap + progress logs)
"""Content-based filtering using TF-IDF item text with candidate cap.

This module implements a TF‑IDF‑based content‑based recommender (CBF). It
constructs a TF‑IDF representation for each item using textual metadata, builds user profiles by
averaging the TF‑IDF vectors of items the user has rated above a threshold,
and ranks candidate items for each user using cosine similarity between the
user profile vector and each item vector. Unseen items with the highest
similarity scores are recommended.  The code also computes offline metrics
(precision@K, recall@K, nDCG@K and hit‑rate@K) on validation and test sets.

refrences 

Lecture 3 – Content‑Based Filtering (CBF) (slides on TF‑IDF and user profiles).  
  These slides introduce representing items via their textual and
  categorical features, using bag‑of‑words or TF‑IDF to obtain feature
  vectors.  They explain that early CBF methods rely on TF‑IDF vectors and
  build user profiles by averaging the vectors of items the user has liked
  (often weighting by rating).  Recommendations are produced by ranking
  unseen items according to cosine similarity between the user profile and
  item vectors.  Our implementation follows this pipeline: feature
  extraction with `TfidfVectorizer`, user profile aggregation, cosine
  similarity scoring and ranking.
  
  Analytics Vidhya (2015), “Beginners Guide to Content‑Based Recommender
   Systems.”  
   This guide explains why TF‑IDF is popular in text‑based recommenders: it
   down‑weights very common words (“the” appears frequently but carries little
   meaning) and emphasises more informative terms; log scaling is used to
   dampen the effect of high term frequencies. After
   computing TF‑IDF scores, the guide advocates using a vector space model
   to compute similarity: each item is stored as a vector of attributes, a
   user profile is created by combining the vectors of liked items, and
   similarity is measured by the cosine of the angle between vectors. These
   principles justify our use of TF‑IDF to build item vectors and user
   profiles, and our use of cosine similarity for ranking.  
   URL: https://www.analyticsvidhya.com/blog/2015/08/beginners-guide-learn-content-based-recommender-systems/


Pipeline:
  1) Load standardized train/val/test ratings and items.
  2) Ensure a text field per item (compose from Title/Categories if missing).
  3) Fit a TF-IDF vectorizer over item text.
  4) Build user profiles from liked items (rating ≥ threshold) and score
     candidates via cosine similarity, excluding already seen items.
  5) Emit top-K recommendations and evaluate precision/recall/nDCG/hit-rate.

Outputs:
  - <outdir>/val_recs_cbf.csv, <outdir>/test_recs_cbf.csv
  - <outdir>/cbf_metrics.json with params and Top-K metrics.

Why these libraries:
  pandas: robust CSV I/O, merging, grouping; common with other project modules.
    Alternative: polars (faster) omitted to minimize dependencies and preserve sklearn interop.
  numpy: vector math for weights and masking. Python lists would be slower and verbose.
  scikit-learn: TfidfVectorizer and normalize are reliable baselines; avoids bespoke text featurizers.
    Alternative: spaCy or gensim skipped to keep memory small and config surface simple.
  scipy.sparse: CSR matrices make TF-IDF and dot products memory-feasible on large catalogs.


Alternatives considered:
  - BM25-like weighting or character n-grams; deferred to keep compute bounded.
  - TF-IDF on title only vs title+categories; current compose prioritizes richer text automatically.
"""



# Collaborative Filtering with item-kNN (cosine) using scikit-learn.

"""Collaborative Filtering with item-kNN (cosine) using scikit-learn.

This module implements a memory‑based collaborative filtering (CF) model that
recommends items by measuring similarities between items in the user–item
interaction matrix.  It loads train/validation/test splits, builds a sparse
user–item rating matrix, fits a k‑nearest neighbours index over the items
(using cosine distance) via scikit‑learn’s `NearestNeighbors`, and then
computes item‑to‑item similarity scores.  For each user it scores candidate
items by aggregating (similarity × user rating) over the user’s rated items
and ranks unseen items accordingly.  The script outputs top‑K recommendations
and computes offline metrics (precision@K, recall@K, nDCG@K and hit‑rate@K)
on validation and test splits.

refrences 

Lecture 4 & 5 – Collaborative Filtering (CF) parts 1–2.  
  These lectures introduce memory‑based CF and describe both user‑based and
  item‑based approaches.  The slides on “Item‑item CF algorithm” and
  “Weighted sum of neighbour ratings” explain that item‑based CF builds a
  similarity matrix between items (using cosine or Pearson similarity) and
  predicts a user’s rating for an item by computing a weighted average of the
  user’s ratings on similar items.  Our implementation follows this algorithm:
  we compute item–item similarities via cosine similarity, select the *k*
  nearest neighbours per item, and use a weighted sum of the user’s ratings to
  score candidates.
  
  GeeksforGeeks (2024), “Item‑to‑Item Based Collaborative Filtering.”  
   This article describes the steps of item‑based CF: (a) compute similarity
   between all item pairs—most commonly using cosine similarity—and
   provides the formula for cosine similarity:contentReference,
   (b) generate predictions by taking a weighted sum of the user’s ratings on
   similar items divided by the sum of the similarities.  
   These equations justify our use of cosine similarity and the weighted‑sum
   scoring function.  
   URL: https://www.geeksforgeeks.org/machine-learning/item-to-item-based-collaborative-filtering/

   GeeksforGeeks (2025), “Recommender Systems using KNN.”
    This article outlines the construction of KNN-based recommender systems by (a) collecting and preprocessing 
    user-item interactions to form a user-item matrix, (b) computing similarity scores between users or items—most 
    commonly using cosine similarity—and (c) finding the k-nearest neighbors for each user or item by sorting the similarity scores. 
    The system then generates recommendations by aggregating the preferences or ratings of these nearest neighbors 
    and recommending items with the highest predicted ratings. The article offers practical code examples 
    using scikit-learn's NearestNeighbors and emphasizes the importance of matrix normalization and efficiency considerations for sparse datasets. 
    These steps and principles support our use of a cosine similarity-based kNN model for item-based collaborative filtering in this project.
    URL: https://www.geeksforgeeks.org/machine-learning/recommender-systems-using-knn/

  Futureweb AI (2025), “Collaborative Filtering‑Based Recommender Systems: A Deep Dive.”  
   This blog explains that item‑based CF computes item–item similarity (using
   metrics such as cosine or Pearson), builds an item similarity matrix, and
   generates recommendations by looking up similar items for each item a user
   has interacted with:contentReference. It notes that item‑based CF is
   often preferred in large‑scale systems because item relationships remain
   more stable over time than user relationships, making it more efficient
  :contentReference. These insights support our choice of an item‑k‑NN
   model for scalability.  
   URL: https://futurewebai.com/blogs/collaborative-filtering-based-recommendation

  
Pipeline:
  1) Load and lightly clean train/val/test CSVs.
  2) Fit an item-kNN model (cosine similarity, brute force).
  3) Generate per-user Top-K recommendations for val/test.
  4) Compute precision@K, recall@K, nDCG@K, hit-rate@K.
  5) Save recommendation files and a metrics JSON.

Outputs:
  <outdir>/val_recs_knn_sklearn.csv
  <outdir>/test_recs_knn_sklearn.csv
  <outdir>/cf_sklearn_metrics.json

Why these libraries:
  pandas: reliable CSV I/O and grouping; consistent with other modules.
  numpy: numeric ops and array handling during evaluation and scoring.
  scipy.sparse: CSR matrices for user–item data keep memory usage low.
  scikit-learn: NearestNeighbors provides a tested brute-force cosine kNN.

Limitations:
   Cold-start items and users cannot be recommended or evaluated.
   Raw ratings used directly; no mean-centering or normalization per user/item.
"""



# Late-fusion hybrid recommender for Top-K ranking.
"""
src/hybrid_fusion.py — late-fusion hybrid for Top-K
Late-fusion hybrid recommender for Top-K ranking.



This module implements a weighted hybrid recommender that combines the

scores of a collaborative‐filtering (CF) model and a content‐based

filtering (CBF) model to produce a single ranked list of items for each user.  

It reads precomputed CF and CBF recommendation files, z‑normalises the scores per user, computes a

log‑scaled popularity score from the training data, and then fuses the

normalised CF scores, CBF scores and popularity weights using a linear

combination: `hybrid_score = w_cf * z_cf + w_cbf * z_cbf + w_pop *

z_pop`. The script performs a small grid search on validation data to

tune the weights `(w_cf, w_cbf, w_pop)` and outputs the resulting

top‑K recommendations and evaluation metrics.






refrences 


Lecture 6 – Hybrid recommender systems (course slides 6–9) discusses

  different hybrid strategies such as weighted (linear) hybrids, switching,

  mixed and cascade hybrids.  Our implementation corresponds to a


  *weighted hybrid*, where the outputs of CF and CBF models are combined


  weighted hybrid, where the outputs of CF and CBF models are combined

  using a weighted average; this design is explicitly contrasted against

  switching or cascade hybrids in the lecture. 

  we also took inspiration from Lecture 4/5 and 3. Collaborative Filtering and Content-Based Filtering.



Marketsy Blog (2024), “Hybrid Recommender Systems: Beginner’s Guide.”

   The article enumerates several hybrid strategies and explains that a

   weighted hybrid combines the outputs of collaborative and content‑based

   models using weighted averages; the importance of each model can be

   adjusted based on its performance:contentReference. For example,

   if the CF model yields more accurate recommendations for a user, its

   output can be given a higher weight. 

   URL: https://marketsy.ai/blog/hybrid-recommender-systems-beginners-guide



Milvus AI Quick Reference (2025), “How do you combine collaborative and

   content‑based methods effectively?”  

   This reference states that hybrid recommender systems blend the outputs

   of CF (which relies on user–item interactions) and CBF (which uses

   item features) to leverage the strengths of both methods.

   It notes that a common strategy is to compute recommendations from both

   models separately and then combine them using weighted averages:contentReference[oaicite:3]{index=3};

   for instance, a movie recommender might use 60 % CF and 40 % CB scores

   to balance popularity and personal preferences.  

   URL: https://milvus.io/ai-quick-reference/how-do-you-combine-collaborative-and-contentbased-methods-effectively



Workflow:

  1) Load CF and CBF recommendation files for val/test.

  2) Standardize scores per user (z-normalization) to make sources comparable.

  3) Map item popularity (log-scaled, globally z-normalized).

  4) Fuse with weights (w_cf, w_cbf, w_pop); optionally tune on validation.

  5) Rank per user, save fused recs, and report Top-K metrics.



Inputs (CSV expectations):

  - Recs: columns [user_id, item_id, score] and/or [rank].

  - Splits: ratings with [user_id, item_id, rating] for relevance ≥ threshold.



Outputs:

  - <outdir>/val_recs_hybrid.csv, <outdir>/test_recs_hybrid.csv

  - <outdir>/hybrid_metrics.json and metrics printed to stdout.

"""