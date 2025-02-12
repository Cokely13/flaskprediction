# import os
# import psycopg2
# import pandas as pd

# def load_data_from_db():
#     """
#     Reads the `DATABASE_URL` env var (if set),
#     or defaults to a local Postgres connection string.
#     """
#     db_url = os.environ.get("DATABASE_URL") or "postgres://localhost:5432/popcornpair"

#     # If you're on Heroku or a service that requires SSL,
#     # and your db.js sets `ssl: { rejectUnauthorized: false }`,
#     # then you might do:
#     #   conn = psycopg2.connect(db_url, sslmode='require')
#     # But let's do the simplest version first:

#     conn = psycopg2.connect(db_url)

#     # Example query: your table might be "UserMovies" with columns userId, movieId, rating, etc.
#     query = """
#         SELECT "userId", "movieId", "rating", "status", "dateWatched", "predictedRating"
#         FROM "user_movies"
#         WHERE "rating" IS NOT NULL
#     """
#     df = pd.read_sql(query, conn)
#     conn.close()

#     # Convert to string if needed
#     df['userId'] = df['userId'].astype(str)
#     df['movieId'] = df['movieId'].astype(str)

#     return df

# db_loader.py

# import os
# import psycopg2
# import pandas as pd

# def load_data_from_db():
#     """
#     Loads all the info needed for both rule-based and CF:
#       - user_movies joined with movies => df_main
#          (contains userId, movieId, rating, criticScore, userRating, etc.)
#       - friend relationships => df_friends
#          (contains userId, friendId where status='accepted')

#     Returns: (df_main, df_friends)
#     """

#     print("[DEBUG] Entering load_data_from_db()...")

#     db_url = os.environ.get("DATABASE_URL") or "postgres://localhost:5432/popcornpair"
#     print(f"[DEBUG] Using DB URL: {db_url}")

#     # 1) Connect to DB
#     conn = psycopg2.connect(db_url)

#     # 2) Query user_movies joined with movies
#     #    This ensures we get 'criticScore' and 'userRating' from the 'movies' table.
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

#     # Convert userId/movieId to string if needed
#     df_main['userId'] = df_main['userId'].astype(str)
#     df_main['movieId'] = df_main['movieId'].astype(str)
#     df_friends['userId'] = df_friends['userId'].astype(str)
#     df_friends['friendId'] = df_friends['friendId'].astype(str)

#     print("[DEBUG] Exiting load_data_from_db()...")
#     return df_main, df_friends

import os
import psycopg2
import pandas as pd

def load_data_from_db():
    """
    Loads enriched data:
      - df_main: user_movies joined with movies (includes criticScore, userRating, etc.)
      - df_friends: accepted friend relationships
    Returns: (df_main, df_friends)
    """
    print("[DEBUG] Entering load_data_from_db()...")

    db_url = os.environ.get("DATABASE_URL") or "postgres://localhost:5432/popcornpair"
    print(f"[DEBUG] Using DB URL: {db_url}")
    conn = psycopg2.connect(db_url)

    # Query main data (join user_movies with movies)
    query_main = """
    SELECT um."userId",
           um."movieId",
           um."rating",
           um."status",
           um."dateWatched",
           um."predictedRating",
           m."criticScore",
           m."userRating"
    FROM "user_movies" AS um
    JOIN "movies" AS m
         ON um."movieId" = m.id
    """
    df_main = pd.read_sql(query_main, conn)
    print("[DEBUG] df_main columns:", df_main.columns.tolist())
    print("[DEBUG] df_main sample:\n", df_main.head(5))

    # Query friend relationships (only accepted friends)
    query_friends = """
    SELECT "userId", "friendId"
    FROM "friends"
    WHERE "status" = 'accepted'
    """
    df_friends = pd.read_sql(query_friends, conn)
    print("[DEBUG] df_friends columns:", df_friends.columns.tolist())
    print("[DEBUG] df_friends sample:\n", df_friends.head(5))

    conn.close()

    # Convert IDs to string for consistency
    df_main['userId'] = df_main['userId'].astype(str)
    df_main['movieId'] = df_main['movieId'].astype(str)
    df_friends['userId'] = df_friends['userId'].astype(str)
    df_friends['friendId'] = df_friends['friendId'].astype(str)

    print("[DEBUG] Exiting load_data_from_db()...")
    return df_main, df_friends
