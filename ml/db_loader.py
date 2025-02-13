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
