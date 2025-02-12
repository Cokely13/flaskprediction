# ########################################
# # ml_model.py
# ########################################
# import pandas as pd
# import numpy as np

# def load_data():
#     """
#     Load the training_data.csv (created by data_retrieval.js).
#     """
#     try:
#         data = pd.read_csv("training_data.csv")
#         return data
#     except Exception as e:
#         print(f"❌ Error loading dataset: {e}")
#         return None

# def predict_rule_based(user_id, movie_id, data):
#     """
#     Predicts a rating for any (userId, movieId) even if no userMovie row exists.
#     """

#     # 1) Filter rows for this movie
#     movieRows = data[data['movieId'] == movie_id]
#     if movieRows.empty:
#         # fallback: average of global criticScore, userRating
#         globalCritic = data['criticScore'].dropna().mean() / 10
#         globalUserRat = data['userRating'].dropna().mean()
#         baseFallback = ( (globalCritic or 5) + (globalUserRat or 5 ) ) / 2
#         print(f"⚠️ No data found for movie {movie_id} - returning fallback {baseFallback:.2f}")
#         return round(baseFallback, 2)

#     # 2) Critic + userRating for the movie
#     # We'll just take the *first* row for the movie's criticScore, userRating
#     # (they should be the same across all userRows for that movie)
#     critic = movieRows.iloc[0]['criticScore'] or 0
#     userRat = movieRows.iloc[0]['userRating'] or 0

#     # scale critic from 0-100 to 0-10 if needed
#     criticScaled = critic / 10

#     # Base prediction
#     basePrediction = (criticScaled + userRat) / 2

#     # 3) Friend Ratings
#     # among data for this movie, which rows are friends?
#     # We look for friend rows with a valid rating
#     # We do not rely on userMovie row for THIS userId, just the friend
#     # If userFriends are known, we might need friend relationships from data too
#     # For simplicity, assume if userId != currentUserId, they might be a friend
#     # (or you'd need a separate dataset to find actual friend userIds).
#     # We'll do a simpler approach: average all other user ratings for this movie
#     otherUserRows = movieRows[movieRows['userId'] != str(user_id)]  # might be strings, watch out
#     friendRatings = pd.to_numeric(otherUserRows['rating'], errors='coerce').dropna()
#     if not friendRatings.empty:
#         avgFriend = friendRatings.mean()
#         # Weighted approach:
#         # 0.5 * friend + 0.25 * basePrediction + 0.25 * (criticScaled + userRat)
#         basePrediction = 0.5 * avgFriend + 0.25 * basePrediction + 0.25*(criticScaled + userRat)

#     # 4) User Tendency
#     # We look for all data where userId == current user, rating != null
#     userRows = data[(data['userId'] == str(user_id)) & (data['rating'].notna())]
#     if len(userRows) > 3:  # only if they've rated >3 movies
#         # Simple approach: userAvg - globalAvg
#         userAvg = userRows['rating'].mean()
#         globalAvg = data['rating'].dropna().mean()
#         userBias = userAvg - globalAvg
#         basePrediction += userBias

#     # clamp to [1..10]
#     final = max(1, min(10, basePrediction))
#     return round(final, 2)

########################################
# ml_model.py
########################################

# import os
# import psycopg2
# import pandas as pd
# import numpy as np

# def load_enriched_data():
#     """
#     Loads all the info needed to do a rule-based prediction:
#       - user_movies joined with movies => df_main
#          (contains userId, movieId, rating, criticScore, userRating, etc.)
#       - friend relationships => df_friends
#          (contains userId, friendId for accepted friendships)

#     Returns: (df_main, df_friends)
#     """

#     print("[DEBUG] Entering load_enriched_data()...")

#     # 1) Connect to your Postgres DB
#     db_url = os.environ.get("DATABASE_URL") or "postgres://localhost:5432/popcornpair"
#     print(f"[DEBUG] Using DB URL: {db_url}")
#     conn = psycopg2.connect(db_url)



#     # 2) Query user_movies joined with movies
#     #    You might adapt columns as needed.
#     query_main = """
#     SELECT um."userId",
#            um."movieId",
#            um."rating",
#            um."status",
#            um."dateWatched",
#            um."predictedRating",
#            m."criticScore",
#            m."userRating"
#     FROM "user_movies" AS um
#     JOIN "movies" AS m
#          ON um."movieId" = m.id
#     """
#     df_main = pd.read_sql(query_main, conn)
#     print("[DEBUG] df_main columns:", df_main.columns.tolist())
#     print("[DEBUG] df_main sample:\n", df_main.head(5))

#     # 3) Load friend relationships (only accepted)
#     query_friends = """
#     SELECT "userId", "friendId"
#     FROM "friends"
#     WHERE "status" = 'accepted'
#     """
#     df_friends = pd.read_sql(query_friends, conn)
#     print("[DEBUG] df_friends columns:", df_friends.columns.tolist())
#     print("[DEBUG] df_friends sample:\n", df_friends.head(5))

#     conn.close()

#     # Ensure userId, movieId are strings
#     df_main['userId'] = df_main['userId'].astype(str)
#     df_main['movieId'] = df_main['movieId'].astype(str)
#     df_friends['userId'] = df_friends['userId'].astype(str)
#     df_friends['friendId'] = df_friends['friendId'].astype(str)


#     print("[DEBUG] Exiting load_enriched_data()...")
#     return df_main, df_friends


# def predict_rule_based(user_id, movie_id, data, df_friends=None):
#     """
#     Predict a rating for (userId, movieId) using:
#       - criticScore (0-100 scaled to 0-10)
#       - userRating (0-10)
#       - friend ratings (only from accepted friends if df_friends is provided)
#       - user bias (if user has >3 ratings)
#     Exactly like your old CSV-based logic, but using real DB data.

#     :param user_id: The user for whom we predict.
#     :param movie_id: The movie in question.
#     :param data: A DataFrame (df_main) with columns:
#                  [userId, movieId, rating, criticScore, userRating, ...].
#     :param df_friends: A DataFrame of friend relationships, or None.
#     :return: A float rating (1..10).
#     """

#     user_str = str(user_id)
#     movie_str = str(movie_id)

#     print(f"[DEBUG] predict_rule_based() called with user_id={user_id}, movie_id={movie_id}")

#     # 1) Filter to rows for this movie
#     movie_rows = data[data['movieId'] == movie_str]
#     print("[DEBUG] movie_rows shape:", movie_rows.shape)

#     if movie_rows.empty:
#         # fallback to a global average of criticScore & userRating
#         global_critic = data['criticScore'].dropna().mean() / 10  # scale 0..100 to 0..10
#         global_userRat = data['userRating'].dropna().mean()
#         baseFallback = ((global_critic or 5) + (global_userRat or 5)) / 2
#         print(f"⚠️ No data found for movie {movie_id} - returning fallback {baseFallback:.2f}")
#         return round(baseFallback, 2)

#     # 2) criticScore + userRating for the movie
#     #    We'll just take the first row (assuming same across all userRows for that movie)
#     critic = movie_rows.iloc[0]['criticScore'] or 0
#     userRat = movie_rows.iloc[0]['userRating'] or 0
#     criticScaled = critic / 10  # scale from 0..100 → 0..10

#     basePrediction = (criticScaled + userRat) / 2

#     print(f"[DEBUG] criticScaled={criticScaled}, userRat={userRat}, basePrediction={basePrediction}")

#     # 3) Friend Ratings
#     #    We'll look for which *accepted friends* (if df_friends provided) have rated this movie.
#     if df_friends is not None:
#         # find all friendIDs for user
#         # user can appear as userId or friendId in df_friends
#         friend_ids = set()

#         # direct matches
#         user_friends = df_friends[df_friends['userId'] == user_str]['friendId'].tolist()
#         friend_ids.update(user_friends)

#         # reverse matches
#         reverse_friends = df_friends[df_friends['friendId'] == user_str]['userId'].tolist()
#         friend_ids.update(reverse_friends)

#         print(f"[DEBUG] friend_ids for user={user_id} => {friend_ids}")

#         # Now find which of those friends actually rated this movie
#         friend_rows = movie_rows[movie_rows['userId'].isin(friend_ids)]
#         friend_ratings = pd.to_numeric(friend_rows['rating'], errors='coerce').dropna()

#         if not friend_ratings.empty:
#             avgFriend = friend_ratings.mean()

#             print("[DEBUG] avgFriend rating:", avgFriend)
#             # Weighted approach
#             basePrediction = 0.5 * avgFriend + 0.25 * basePrediction + 0.25 * (criticScaled + userRat)
#     else:
#         # fallback: consider all other users as "friends"
#         other_users = movie_rows[movie_rows['userId'] != user_str]
#         friend_ratings = pd.to_numeric(other_users['rating'], errors='coerce').dropna()
#         if not friend_ratings.empty:
#             avgFriend = friend_ratings.mean()
#             basePrediction = 0.5 * avgFriend + 0.25 * basePrediction + 0.25 * (criticScaled + userRat)

#     # 4) User Tendency / Bias
#     #    If user has >3 ratings in data, we shift the base by userAvg - globalAvg
#     user_rows = data[(data['userId'] == user_str) & (data['rating'].notna())]
#     if len(user_rows) > 3:
#         userAvg = user_rows['rating'].mean()
#         globalAvg = data['rating'].dropna().mean()
#         userBias = userAvg - globalAvg
#         basePrediction += userBias

#     # clamp to [1..10]
#     final = max(1, min(10, basePrediction))
#     return round(final, 2)


# def predict_rule_based_fresh(user_id, movie_id):
#     """
#     Convenience function if you want a single call that:
#     1) Loads fresh data from DB
#     2) Calls predict_rule_based

#     Then you don't need to pass a DataFrame around.
#     """
#     df_main, df_friends = load_enriched_data()
#     return predict_rule_based(user_id, movie_id, df_main, df_friends)

########################################
# ml_model.py
########################################
########################################
# ml_model.py
########################################

# import os
# import psycopg2
# import pandas as pd
# import numpy as np

# def load_enriched_data():
#     """
#     Loads all the info needed to do a rule-based prediction:
#       - user_movies joined with movies (df_main)
#          (contains userId, movieId, rating, status, dateWatched, predictedRating, criticScore, userRating)
#       - Friend relationships (df_friends)
#          (contains userId, friendId for accepted friendships)
#     Returns: (df_main, df_friends)
#     """
#     # Here we delegate to our updated db_loader
#     from db_loader import load_data_from_db
#     return load_data_from_db()

# def predict_rule_based(user_id, movie_id, data, df_friends=None):
#     """
#     Predicts a rating for (userId, movieId) using:
#       - criticScore (scaled 0-100 → 0-10)
#       - userRating (0-10)
#       - Friend ratings (if df_friends is provided)
#       - User bias (if the user has >3 ratings)
#     """
#     user_str = str(user_id)
#     movie_str = str(movie_id)

#     print(f"[DEBUG] predict_rule_based() called with user_id={user_id}, movie_id={movie_id}")

#     # 1) Filter to rows for this movie
#     movie_rows = data[data['movieId'] == movie_str]
#     print("[DEBUG] movie_rows shape:", movie_rows.shape)

#     if movie_rows.empty:
#         # Fallback: use global averages
#         global_critic = data['criticScore'].dropna().mean() / 10  # scale to 0-10
#         global_userRat = data['userRating'].dropna().mean()
#         baseFallback = ((global_critic or 5) + (global_userRat or 5)) / 2
#         print(f"⚠️ No data found for movie {movie_id} - returning fallback {baseFallback:.2f}")
#         return round(baseFallback, 2)

#     # 2) Get the first matching row (ensure it's a Series)
#     row = movie_rows.iloc[0]
#     if not isinstance(row, pd.Series):
#         row = pd.Series(row, index=movie_rows.columns)

#     critic = row['criticScore'] or 0
#     userRat = row['userRating'] or 0
#     criticScaled = critic / 10  # convert to scale 0-10
#     basePrediction = (criticScaled + userRat) / 2
#     print(f"[DEBUG] criticScaled={criticScaled}, userRat={userRat}, basePrediction={basePrediction}")

#     # 3) Incorporate friend ratings if provided
#     if df_friends is not None:
#         friend_ids = set()
#         # Direct friend relationships
#         user_friends = df_friends[df_friends['userId'] == user_str]['friendId'].tolist()
#         friend_ids.update(user_friends)
#         # Reverse relationships
#         reverse_friends = df_friends[df_friends['friendId'] == user_str]['userId'].tolist()
#         friend_ids.update(reverse_friends)
#         print(f"[DEBUG] friend_ids for user {user_id}: {friend_ids}")

#         friend_rows = movie_rows[movie_rows['userId'].isin(friend_ids)]
#         friend_ratings = pd.to_numeric(friend_rows['rating'], errors='coerce').dropna()
#         if not friend_ratings.empty:
#             avgFriend = friend_ratings.mean()
#             print("[DEBUG] avgFriend rating:", avgFriend)
#             basePrediction = 0.5 * avgFriend + 0.25 * basePrediction + 0.25 * ((criticScaled + userRat) / 2)
#     else:
#         # Fallback: treat all other users as friends
#         other_users = movie_rows[movie_rows['userId'] != user_str]
#         friend_ratings = pd.to_numeric(other_users['rating'], errors='coerce').dropna()
#         if not friend_ratings.empty:
#             avgFriend = friend_ratings.mean()
#             basePrediction = 0.5 * avgFriend + 0.25 * basePrediction + 0.25 * ((criticScaled + userRat) / 2)

#         #    0.5 * avgFriend + 0.25 * basePrediction + 0.25 * (criticScaled + userRat)

#     # 4) Add user tendency if user has more than 3 ratings
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
#       - criticScore (scaled 0-100 → 0-10) if available,
#       - userRating (0-10) if available,
#       - Friend ratings (if df_friends is provided),
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
#         global_critic = data['criticScore'].dropna().mean() / 10  # scaled to 0-10
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
#         # If both are missing, fallback to global averages.
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
#         criticScaled = critic / 10  # scale from 0-100 to 0-10
#         basePrediction = (criticScaled + userRat) / 2
#         print(f"[DEBUG] criticScaled={criticScaled}, userRat={userRat}, basePrediction={basePrediction}")

#     # 3) Incorporate friend ratings if provided
#     if df_friends is not None:
#         friend_ids = set()
#         # Direct relationships
#         friend_ids.update(df_friends[df_friends['userId'] == user_str]['friendId'].tolist())
#         # Reverse relationships
#         friend_ids.update(df_friends[df_friends['friendId'] == user_str]['userId'].tolist())
#         print(f"[DEBUG] friend_ids for user {user_id}: {friend_ids}")

#         friend_rows = movie_rows[movie_rows['userId'].isin(friend_ids)]
#         friend_ratings = pd.to_numeric(friend_rows['rating'], errors='coerce').dropna()
#         if not friend_ratings.empty:
#             avgFriend = friend_ratings.mean()
#             print("[DEBUG] avgFriend rating:", avgFriend)
#             if critic is None or userRat is None:
#                 # Use equal weighting if one of the values is missing.
#                 basePrediction = 0.5 * avgFriend + 0.5 * basePrediction
#             else:
#                 # Use the average of (criticScaled + userRat) divided by 2 for consistent scale.
#                 basePrediction = 0.5 * avgFriend + 0.25 * basePrediction + 0.25 * ((critic / 10 + userRat) / 2)

#     else:
#         # Fallback: treat all other users as friends
#         other_users = movie_rows[movie_rows['userId'] != user_str]
#         friend_ratings = pd.to_numeric(other_users['rating'], errors='coerce').dropna()
#         if not friend_ratings.empty:
#             avgFriend = friend_ratings.mean()
#             basePrediction = 0.5 * avgFriend + 0.25 * basePrediction + 0.25 * ((critic / 10 + userRat) / 2)

#     # 4) Add user bias if the user has more than 3 ratings
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
    # Delegate to our updated db_loader
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

    # 2) Get the first matching row (ensure it's a Series)
    row = movie_rows.iloc[0]
    if not isinstance(row, pd.Series):
        row = pd.Series(row, index=movie_rows.columns)

    # Check if criticScore or userRating is missing
    critic = row['criticScore'] if pd.notna(row['criticScore']) else None
    userRat = row['userRating'] if pd.notna(row['userRating']) else None

    if critic is None and userRat is None:
        # If both are missing, fall back to global averages.
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

    # 3) Incorporate friend ratings if provided
    if df_friends is not None:
        friend_ids = set()
        # Get direct and reverse friend relationships
        friend_ids.update(df_friends[df_friends['userId'] == user_str]['friendId'].tolist())
        friend_ids.update(df_friends[df_friends['friendId'] == user_str]['userId'].tolist())
        print(f"[DEBUG] friend_ids for user {user_id}: {friend_ids}")

        friend_rows = movie_rows[movie_rows['userId'].isin(friend_ids)]
        friend_ratings = pd.to_numeric(friend_rows['rating'], errors='coerce').dropna()
        if not friend_ratings.empty:
            avgFriend = friend_ratings.mean()
            numFriendRatings = len(friend_ratings)
            # Confidence factor increases with the number of friend ratings, up to full weight at 5 ratings.
            confidence = min(numFriendRatings / 5, 1)
            print(f"[DEBUG] avgFriend rating: {avgFriend} with {numFriendRatings} ratings, confidence: {confidence}")
            if critic is None or userRat is None:
                # When only one value is available, use equal weighting.
                basePrediction = 0.5 * avgFriend + 0.5 * basePrediction
            else:
                # Compute friend blend: 60% friend average, 40% baseline.
                friendBlend = 0.6 * avgFriend + 0.4 * basePrediction
                # Blend the original baseline with the friend blend, weighted by confidence.
                basePrediction = (1 - confidence) * basePrediction + confidence * friendBlend
    else:
        # Fallback: treat all other users as friends
        other_users = movie_rows[movie_rows['userId'] != user_str]
        friend_ratings = pd.to_numeric(other_users['rating'], errors='coerce').dropna()
        if not friend_ratings.empty:
            avgFriend = friend_ratings.mean()
            basePrediction = 0.5 * avgFriend + 0.25 * basePrediction + 0.25 * ((critic / 10 + userRat) / 2)

    # 4) Incorporate user bias if the user has >3 ratings
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
