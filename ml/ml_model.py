# import os
# import psycopg2
# import pandas as pd
# import numpy as np
# import math

# def load_enriched_data():
#     """
#     Loads all the info needed to do a rule-based prediction:
#       - user_movies joined with movies (df_main)
#          (contains userId, movieId, rating, status, dateWatched, predictedRating, criticScore, userRating)
#       - Friend relationships (df_friends)
#          (contains userId, friendId for accepted friendships)
#     Returns: (df_main, df_friends)
#     """
#     # Delegate to our updated db_loader
#     from db_loader import load_data_from_db
#     return load_data_from_db()

# def predict_rule_based(user_id, movie_id, data, df_friends=None):
#     """
#     Predicts a rating for (userId, movieId) using:
#       - criticScore (scaled from 0–100 to 0–10) if available,
#       - userRating (0–10) if available,
#       - Friend ratings (if df_friends is provided) with a confidence factor,
#       - User bias (if the user has >3 ratings)

#     If either criticScore or userRating is missing, the prediction uses only the available value.
#     If both are missing, it falls back to global averages.
#     """
#     user_str = str(user_id)
#     movie_str = str(movie_id)
#     print(f"[DEBUG] predict_rule_based() called with user_id={user_id}, movie_id={movie_id}")

#     # 1) Filter to rows for this movie
#     movie_rows = data[data['movieId'] == movie_str]
#     print("[DEBUG] movie_rows shape:", movie_rows.shape)
#     if movie_rows.empty:
#         # Fallback: use global averages if no data for the movie exists.
#         global_critic = data['criticScore'].dropna().mean() / 10  # scaled to 0–10
#         global_userRat = data['userRating'].dropna().mean()
#         baseFallback = ((global_critic or 5) + (global_userRat or 5)) / 2
#         print(f"⚠️ No data found for movie {movie_id} - returning fallback {baseFallback:.2f}")
#         return round(baseFallback, 2)

#     # 2) Get the first matching row (ensure it's a Series)
#     row = movie_rows.iloc[0]
#     if not isinstance(row, pd.Series):
#         row = pd.Series(row, index=movie_rows.columns)

#     # Check if criticScore or userRating is missing
#     critic = row['criticScore'] if pd.notna(row['criticScore']) else None
#     userRat = row['userRating'] if pd.notna(row['userRating']) else None

#     if critic is None and userRat is None:
#         # If both are missing, fall back to global averages.
#         global_critic = data['criticScore'].dropna().mean() / 10
#         global_userRat = data['userRating'].dropna().mean()
#         basePrediction = ((global_critic or 5) + (global_userRat or 5)) / 2
#         print(f"[DEBUG] Both critic and user rating missing for movie {movie_id} - fallback basePrediction={basePrediction}")
#     elif critic is None:
#         basePrediction = userRat
#         print(f"[DEBUG] Critic missing, using userRat={userRat} as basePrediction")
#     elif userRat is None:
#         basePrediction = critic / 10
#         print(f"[DEBUG] User rating missing, using criticScaled={critic/10} as basePrediction")
#     else:
#         criticScaled = critic / 10  # Scale critic score from 0–100 to 0–10
#         basePrediction = (criticScaled + userRat) / 2
#         print(f"[DEBUG] criticScaled={criticScaled}, userRat={userRat}, basePrediction={basePrediction}")

#     # 3) Incorporate friend ratings if provided
#     if df_friends is not None:
#         friend_ids = set()
#         # Get direct and reverse friend relationships
#         friend_ids.update(df_friends[df_friends['userId'] == user_str]['friendId'].tolist())
#         friend_ids.update(df_friends[df_friends['friendId'] == user_str]['userId'].tolist())
#         print(f"[DEBUG] friend_ids for user {user_id}: {friend_ids}")

#         friend_rows = movie_rows[movie_rows['userId'].isin(friend_ids)]
#         friend_ratings = pd.to_numeric(friend_rows['rating'], errors='coerce').dropna()
#         if not friend_ratings.empty:
#             avgFriend = friend_ratings.mean()
#             numFriendRatings = len(friend_ratings)
#             # Confidence factor increases with the number of friend ratings, up to full weight at 5 ratings.
#             confidence = min(numFriendRatings / 5, 1)
#             print(f"[DEBUG] avgFriend rating: {avgFriend} with {numFriendRatings} ratings, confidence: {confidence}")
#             if critic is None or userRat is None:
#                 # When only one value is available, use equal weighting.
#                 basePrediction = 0.5 * avgFriend + 0.5 * basePrediction
#             else:
#                 # Compute friend blend: 60% friend average, 40% baseline.
#                 friendBlend = 0.6 * avgFriend + 0.4 * basePrediction
#                 # Blend the original baseline with the friend blend, weighted by confidence.
#                 basePrediction = (1 - confidence) * basePrediction + confidence * friendBlend
#     else:
#         # Fallback: treat all other users as friends
#         other_users = movie_rows[movie_rows['userId'] != user_str]
#         friend_ratings = pd.to_numeric(other_users['rating'], errors='coerce').dropna()
#         if not friend_ratings.empty:
#             avgFriend = friend_ratings.mean()
#             basePrediction = 0.5 * avgFriend + 0.25 * basePrediction + 0.25 * ((critic / 10 + userRat) / 2)

#     # 4) Incorporate user bias if the user has >3 ratings
#     user_rows = data[(data['userId'] == user_str) & (data['rating'].notna())]
#     print("[DEBUG] user_rows count for user", user_id, "=", len(user_rows))
#     if len(user_rows) > 3:
#         userAvg = user_rows['rating'].mean()
#         globalAvg = data['rating'].dropna().mean()
#         userBias = userAvg - globalAvg
#         basePrediction += userBias
#         print(f"[DEBUG] userAvg={userAvg}, globalAvg={globalAvg}, userBias={userBias}")

#     final = max(1, min(10, basePrediction))
#     print(f"[DEBUG] final rule-based rating = {final}")
#     return round(final, 2)

# def predict_rule_based_fresh(user_id, movie_id):
#     """
#     Convenience function:
#       1) Loads fresh enriched data from the DB.
#       2) Calls predict_rule_based.
#     """
#     df_main, df_friends = load_enriched_data()
#     return predict_rule_based(user_id, movie_id, df_main, df_friends)


import os
import psycopg2
import pandas as pd
import numpy as np
import math

def load_enriched_data():
    """
    Loads all the info needed to do a rule-based prediction:
      - user_movies joined with movies (df_main)
         (contains userId, movieId, rating, status, dateWatched, predictedRating, criticScore, userRating)
      - Friend relationships (df_friends)
         (contains userId, friendId for accepted friendships)
    Returns: (df_main, df_friends)
    """
    from db_loader import load_data_from_db
    return load_data_from_db()

def predict_rule_based(user_id, movie_id, data, df_friends=None):
    """
    Predicts a rating for (userId, movieId) using:
      - criticScore (scaled from 0–100 to 0–10) if available,
      - userRating (0–10) if available,
      - Friend ratings (if df_friends is provided) with a confidence factor,
      - User bias (if the user has >3 ratings)

    If either criticScore or userRating is missing, the prediction uses only the available value.
    If both are missing, it falls back to global averages.
    """
    user_str = str(user_id)
    movie_str = str(movie_id)
    print(f"[DEBUG] predict_rule_based() called with user_id={user_id}, movie_id={movie_id}")

    # 1) Filter to rows for this movie
    movie_rows = data[data['movieId'] == movie_str]
    print("[DEBUG] movie_rows shape:", movie_rows.shape)
    if movie_rows.empty:
        # Fallback: use global averages if no data for the movie exists.
        global_critic = data['criticScore'].dropna().mean() / 10  # scaled to 0–10
        global_userRat = data['userRating'].dropna().mean()
        baseFallback = ((global_critic or 5) + (global_userRat or 5)) / 2
        print(f"⚠️ No data found for movie {movie_id} - returning fallback {baseFallback:.2f}")
        return round(baseFallback, 2)

    # 2) Prefer the row with no user_movie data (i.e. where userId is empty), if available.
    movie_rows_no_um = movie_rows[movie_rows['userId'] == '']
    if not movie_rows_no_um.empty:
        row = movie_rows_no_um.iloc[0]
        print("[DEBUG] Using movie row with no user_movie data")
    else:
        row = movie_rows.iloc[0]
        print("[DEBUG] Using first available row from movie_rows")

    # Extract movie-level values from the row
    critic = row['criticScore'] if pd.notna(row['criticScore']) else None
    userRat = row['userRating'] if pd.notna(row['userRating']) else None

    # 3) Compute the base prediction from movie-level data.
    if critic is None and userRat is None:
        global_critic = data['criticScore'].dropna().mean() / 10
        global_userRat = data['userRating'].dropna().mean()
        basePrediction = ((global_critic or 5) + (global_userRat or 5)) / 2
        print(f"[DEBUG] Both critic and user rating missing for movie {movie_id} - fallback basePrediction={basePrediction}")
    elif critic is None:
        basePrediction = userRat
        print(f"[DEBUG] Critic missing, using userRat={userRat} as basePrediction")
    elif userRat is None:
        basePrediction = critic / 10
        print(f"[DEBUG] User rating missing, using criticScaled={critic/10} as basePrediction")
    else:
        criticScaled = critic / 10  # Scale critic score from 0–100 to 0–10
        basePrediction = (criticScaled + userRat) / 2
        print(f"[DEBUG] criticScaled={criticScaled}, userRat={userRat}, basePrediction={basePrediction}")

    # 4) Incorporate friend ratings if provided
    if df_friends is not None:
        friend_ids = set()
        # Get friend relationships: both direct and reverse
        friend_ids.update(df_friends[df_friends['userId'] == user_str]['friendId'].tolist())
        friend_ids.update(df_friends[df_friends['friendId'] == user_str]['userId'].tolist())
        print(f"[DEBUG] friend_ids for user {user_id}: {friend_ids}")

        # Note: Here we use movie_rows (all rows for the movie) so that if user_movies exist, we capture friend ratings.
        friend_rows = movie_rows[movie_rows['userId'].isin(friend_ids)]
        friend_ratings = pd.to_numeric(friend_rows['rating'], errors='coerce').dropna()
        if not friend_ratings.empty:
            avgFriend = friend_ratings.mean()
            numFriendRatings = len(friend_ratings)
            # Confidence factor increases with the number of friend ratings, up to full weight at 5 ratings.
            confidence = min(numFriendRatings / 5, 1)
            print(f"[DEBUG] avgFriend rating: {avgFriend} with {numFriendRatings} ratings, confidence: {confidence}")
            if critic is None or userRat is None:
                basePrediction = 0.5 * avgFriend + 0.5 * basePrediction
            else:
                # Compute friend blend: 60% friend average, 40% baseline.
                friendBlend = 0.6 * avgFriend + 0.4 * basePrediction
                basePrediction = (1 - confidence) * basePrediction + confidence * friendBlend
    else:
        # Fallback: treat all other users as friends
        other_users = movie_rows[movie_rows['userId'] != user_str]
        friend_ratings = pd.to_numeric(other_users['rating'], errors='coerce').dropna()
        if not friend_ratings.empty:
            avgFriend = friend_ratings.mean()
            basePrediction = 0.5 * avgFriend + 0.25 * basePrediction + 0.25 * ((critic / 10 + userRat) / 2)

    # 5) Incorporate user bias if the user has more than 3 ratings.
    user_rows = data[(data['userId'] == user_str) & (data['rating'].notna())]
    print("[DEBUG] user_rows count for user", user_id, "=", len(user_rows))
    if len(user_rows) > 3:
        userAvg = user_rows['rating'].mean()
        globalAvg = data['rating'].dropna().mean()
        userBias = userAvg - globalAvg
        basePrediction += userBias
        print(f"[DEBUG] userAvg={userAvg}, globalAvg={globalAvg}, userBias={userBias}")

    final = max(1, min(10, basePrediction))
    print(f"[DEBUG] final rule-based rating = {final}")
    return round(final, 2)

def predict_rule_based_fresh(user_id, movie_id):
    """
    Convenience function:
      1) Loads fresh enriched data from the DB.
      2) Calls predict_rule_based.
    """
    df_main, df_friends = load_enriched_data()
    return predict_rule_based(user_id, movie_id, df_main, df_friends)
