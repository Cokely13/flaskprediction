####################################################
# pure_cf.py
####################################################
import pandas as pd
import numpy as np
from .db_loader import load_data_from_db

# def load_data(csv_path="training_data.csv"):
#     """
#     Loads user-movie-rating data from CSV.
#     Expects columns: userId, movieId, rating (numeric).
#     """
#     df = pd.read_csv(csv_path)
#     # Drop rows with missing critical values
#     df.dropna(subset=['userId', 'movieId', 'rating'], inplace=True)
#     return df

def user_based_cf_predict(
    df: pd.DataFrame,
    user_id: str,
    movie_id: str,
    k: int = 5
) -> float:
    """
    A pure-Python user-based collaborative filter prediction.

    :param df: DataFrame with columns [userId, movieId, rating].
    :param user_id: The target user's ID (string or int).
    :param movie_id: The target movie's ID (string or int).
    :param k: Number of neighbors to consider.
    :return: Predicted rating as a float. If no data, returns a fallback (global average).

    Steps:
    1) Pivot the data into a user-item rating matrix.
    2) Compute average rating per user to center ratings.
    3) Cosine similarity between the target user and other users.
    4) Find the top-k neighbors who have rated this movie.
    5) Weighted average of the neighbors' deviations from their mean.
    """

    # Ensure user_id/movie_id are consistent types
    # If your CSV has them as int, you might cast to str or keep as int. Just be consistent.
    df['userId'] = df['userId'].astype(str)
    df['movieId'] = df['movieId'].astype(str)
    user_id = str(user_id)
    movie_id = str(movie_id)

    # 1) Pivot to create user x item matrix (some cells may be NaN)
    rating_matrix = df.pivot_table(index='userId', columns='movieId', values='rating')

    # If the user or movie is missing entirely, fallback to a global average
    if user_id not in rating_matrix.index and movie_id not in rating_matrix.columns:
        return df['rating'].mean()  # global average

    # 2) Compute mean rating per user. We'll use this to center ratings.
    user_mean = rating_matrix.mean(axis=1)

    # 3) "Center" the rating matrix by subtracting each user's mean rating
    rating_centered = rating_matrix.sub(user_mean, axis=0)

    # We also need the target user's centered vector
    # If the user doesn't exist in the matrix, fallback to global average
    if user_id not in rating_centered.index:
        return df['rating'].mean()

    target_vector = rating_centered.loc[user_id].fillna(0)

    # 4) Cosine similarity between the target user and each other user
    def cosine_sim(row):
        row_filled = row.fillna(0)
        denom = (np.linalg.norm(row_filled) * np.linalg.norm(target_vector))
        if denom == 0:
            return 0.0
        return np.dot(row_filled, target_vector) / denom

    sim_scores = rating_centered.apply(cosine_sim, axis=1)

    # We don't want the user to be in their own neighbor list
    sim_scores.drop(user_id, errors='ignore', inplace=True)

    # 5) Identify the top-k neighbors (by similarity score) who have rated this movie
    # We only consider neighbors who actually have a rating for this movie
    if movie_id not in rating_matrix.columns:
        # The movie doesn't exist in the pivot table => fallback
        return df['rating'].mean()

    neighbors_that_rated = rating_matrix[movie_id].dropna().index  # which users have a rating
    # Intersection of those neighbors with the sim_scores index
    valid_neighbors = neighbors_that_rated.intersection(sim_scores.index)

    # If no neighbor has rated this movie, fallback to the user's average rating
    if len(valid_neighbors) == 0:
        return user_mean.get(user_id, df['rating'].mean())

    # Filter similarity scores to valid neighbors only
    sim_scores_valid = sim_scores[valid_neighbors]
    # Sort descending by similarity
    top_neighbors = sim_scores_valid.nlargest(k)

    # Weighted average
    numerator, denominator = 0.0, 0.0

    for neighbor_id, sim_val in top_neighbors.items():
        # neighbor's rating for this movie
        neighbor_rating = rating_matrix.at[neighbor_id, movie_id]
        # neighbor's mean rating
        neighbor_mean = user_mean[neighbor_id]
        # Weighted by similarity
        numerator += sim_val * (neighbor_rating - neighbor_mean)
        denominator += abs(sim_val)

    # If denominator is 0, fallback to target user's mean or global mean
    if denominator == 0:
        return user_mean.get(user_id, df['rating'].mean())

    # Final predicted rating = user_mean + weighted avg of neighbor deviations
    predicted = user_mean[user_id] + (numerator / denominator)
    return float(predicted)

if __name__ == "__main__":
    # Quick test
    df_ratings = load_data_from_db()
    user = "123"
    movie = "456"
    predicted_rating = user_based_cf_predict(df_ratings, user, movie, k=5)
    print(f"Predicted rating for user {user}, movie {movie}: {predicted_rating:.2f}")
