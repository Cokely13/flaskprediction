from .db_loader import load_data_from_db  # Using our enriched loader if you prefer, or load_enriched_data from ml_model.py.
from .ml_model import predict_rule_based
from .pure_cf import user_based_cf_predict

THRESHOLD = 10

def hybrid_predict(user_id, movie_id):
    """
    Decides whether to use rule-based or CF based on how many ratings the user has.
    Loads enriched data from the DB.
    Returns (predicted_rating, approach_str).
    """
    # Use the enriched loader to get all required columns
    df_main, df_friends = load_data_from_db()

    user_str = str(user_id)
    user_ratings_count = df_main[
        (df_main['userId'] == user_str) & (df_main['rating'].notna())
    ].shape[0]

    print(f"[DEBUG] user_ratings_count for user {user_id} = {user_ratings_count}")

    if user_ratings_count < THRESHOLD:
        rating = predict_rule_based(user_id, movie_id, df_main, df_friends)
        print(f"[DEBUG] Returning rule-based rating = {rating}")
        return rating, "rule-based"
    else:
        rating = user_based_cf_predict(df_main, user_id, movie_id, k=5)
        print(f"[DEBUG] Returning CF rating = {rating}")
        return rating, "machine-learning"
