Exploratory Data Analysis (EDA)

Perform deeper data exploration to understand the dataset’s patterns and potential for modeling:

    Statistical overview: Plot rating distribution (e.g., histogram of reviewscore) and compute mean/variance.

    User and item activity: Visualize number of reviews per user/book to check for sparsity.

    Temporal analysis: If timestamps are usable, explore rating trends over time.

    Text analysis (optional): Analyze reviewtext sentiment or word frequency to identify implicit user preferences.

Use libraries like matplotlib, seaborn, and wordcloud for visual insights.


Feature Engineering

Enhance the dataset to support better recommendations:

    Generate numerical features such as average_rating_per_book and average_rating_per_user.

    Process text data with NLP (e.g., TF-IDF, sentiment scores from reviewtext).

    Encode categorical data (e.g., user and book IDs using integer mapping).

Prepare for Recommender System Modeling

Decide on approach and prepare the data accordingly:

    Collaborative Filtering – Build a user-item matrix and split into train/test sets.

        Try algorithms: Matrix Factorization (e.g., SVD) or K-Nearest Neighbors (user-based or item-based).

    Content-Based – Use features from reviewtext or Title to compute item similarity.

    Hybrid Models – Combine both using weighted averaging or stacking methods.

Evaluation Planning

Define the evaluation strategy before modeling:

    Use metrics like RMSE, MAE (for rating prediction) or Precision@k, Recall@k (for top-N recommendation).

    Consider performing cross-validation for robustness checks.



Lecture 3: TF: noen reviews veier mer en andre



Report Integration

Document the following for your project report:

    Problem motivation and research question

    Description of dataset and cleaning process

    EDA findings with visualizations

    Type of recommender chosen and rationale

    Planned modeling approach and evaluation strategy



