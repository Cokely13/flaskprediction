# import pandas as pd
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from ml_model import load_data, predict_rating  # from the code above

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})

# # Load dataset once at startup
# dataset = load_data()
# if dataset is None:
#     print("❌ Could not load training_data.csv. Predictions may fail.")

# @app.route('/api/health', methods=['GET'])
# def health_check():
#     return jsonify({"status": "OK"}), 200

# @app.route('/api/predict-rating', methods=['POST'])
# def predict_rating_endpoint():
#     try:
#         if dataset is None:
#             return jsonify({"error": "Dataset not loaded"}), 500

#         data = request.json
#         if not data or 'userId' not in data or 'movieId' not in data:
#             return jsonify({'error': 'Expected "userId" and "movieId"'}), 400

#         user_id = data['userId']
#         movie_id = data['movieId']

#         # Call rule-based function
#         rating = predict_rating(user_id, movie_id, dataset)
#         return jsonify({"predictedRating": rating})

#     except Exception as e:
#         print(f"Error in /api/predict-rating: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)


# app.py
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# from db_loader import load_data_from_db

# from hybrid_predict import hybrid_predict
# # or from pure_cf import load_data if you're loading data from there
# # from pure_cf import load_data

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})

# # Load dataset at startup
# # df_ratings = load_data("training_data.csv")

# @app.route("/api/predict-rating", methods=["POST"])
# def predict_rating_endpoint():
#     try:
#         data = request.json
#         user_id = data.get("userId")
#         movie_id = data.get("movieId")

#         df = load_data_from_db()

#         if not user_id or not movie_id:
#             return jsonify({"error": "Missing userId or movieId"}), 400

#         # Use the hybrid predict function
#         predicted_rating, approach = hybrid_predict(user_id, movie_id, df)

#         # Return both the rating and the approach
#         return jsonify({
#             "predictedRating": round(predicted_rating, 2),
#             "approachUsed": approach
#         }), 200

#     except Exception as e:
#         print(f"Error in /api/predict-rating: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from hybrid_predict import hybrid_predict

# from flask_cors import CORS

# app = Flask(__name__)

# CORS(app, resources={r"/*": {"origins": "*"}})

# @app.route("/api/predict-rating", methods=["POST"])
# def predict_rating_endpoint():
#     try:
#         data = request.json
#         user_id = data.get("userId")
#         movie_id = data.get("movieId")
#         if not user_id or not movie_id:
#             return jsonify({"error": "Missing userId or movieId"}), 400

#         predicted_rating, approach = hybrid_predict(user_id, movie_id)
#         print(f"[DEBUG] final predicted_rating={predicted_rating}, approach={approach}")

#         return jsonify({
#             "predictedRating": round(predicted_rating, 2),
#             "approachUsed": approach
#         }), 200

#     except Exception as e:
#         print(f"Error in /api/predict-rating: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from hybrid_predict import hybrid_predict

# app = Flask(__name__)

# # This applies CORS to all routes with default settings (allowing any origin)
# # CORS(app, supports_credentials=True)
# CORS(app, resources={r"/*": {"origins": [
#     "https://popcornpair-6403c0694200.herokuapp.com",
#     "http://localhost:8080"
# ]}})

# @app.route("/api/predict-rating", methods=["POST", "OPTIONS"])
# def predict_rating_endpoint():
#     # Flask-CORS should automatically handle OPTIONS requests
#     # if request.method == "OPTIONS":
#     #     return jsonify({}), 200

#     # try:
#     #     data = request.json
#     #     user_id = data.get("userId")
#     #     movie_id = data.get("movieId")
#     #     if not user_id or not movie_id:
#     #         return jsonify({"error": "Missing userId or movieId"}), 400

#     #     predicted_rating, approach = hybrid_predict(user_id, movie_id)
#     #     print(f"[DEBUG] final predicted_rating={predicted_rating}, approach={approach}")

#     #     return jsonify({
#     #         "predictedRating": round(predicted_rating, 2),
#     #         "approachUsed": approach
#     #     }), 200

#     # except Exception as e:
#     #     print(f"Error in /api/predict-rating: {str(e)}")
#     #     return jsonify({"error": str(e)}), 500


#     try:
#         data = request.json
#         user_id = data.get("userId")
#         movie_id = data.get("movieId")
#         if not user_id or not movie_id:
#             return jsonify({"error": "Missing userId or movieId"}), 400

#         predicted_rating, approach = hybrid_predict(user_id, movie_id)
#         print(f"[DEBUG] final predicted_rating={predicted_rating}, approach={approach}")

#         return jsonify({
#             "predictedRating": round(predicted_rating, 2),
#             "approachUsed": approach
#         }), 200

#     except Exception as e:
#         print(f"Error in /api/predict-rating: {str(e)}")
#         return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     app.run(debug=True)

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from hybrid_predict import hybrid_predict

# app = Flask(__name__)

# # Enable CORS for specified origins
# CORS(app, resources={r"/*": {"origins": [
#     "https://popcornpair-6403c0694200.herokuapp.com",
#     "http://localhost:8080"
# ]}})

# # Only allow POST; Flask-CORS will automatically handle OPTIONS requests
# @app.route("/api/predict-rating", methods=["POST"])
# def predict_rating_endpoint():
#     try:
#         data = request.json
#         user_id = data.get("userId")
#         movie_id = data.get("movieId")
#         if not user_id or not movie_id:
#             return jsonify({"error": "Missing userId or movieId"}), 400

#         predicted_rating, approach = hybrid_predict(user_id, movie_id)
#         print(f"[DEBUG] final predicted_rating={predicted_rating}, approach={approach}")

#         return jsonify({
#             "predictedRating": round(predicted_rating, 2),
#             "approachUsed": approach
#         }), 200

#     except Exception as e:
#         print(f"Error in /api/predict-rating: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)


# from flask import Flask, request, jsonify
# from flask_cors import CORS, cross_origin
# from hybrid_predict import hybrid_predict

# app = Flask(__name__)

# # Enable CORS for specified origins
# CORS(app, resources={r"/*": {"origins": [
#     "https://popcornpair-6403c0694200.herokuapp.com",
#     "http://localhost:8080"
# ]}})

# @app.route("/api/predict-rating", methods=["POST", "OPTIONS"])
# @cross_origin(origins=[
#     "https://popcornpair-6403c0694200.herokuapp.com",
#     "http://localhost:8080"
# ])
# def predict_rating_endpoint():
#     # Explicitly handle OPTIONS requests
#     if request.method == "OPTIONS":
#         return jsonify({}), 200

#     try:
#         data = request.json
#         user_id = data.get("userId")
#         movie_id = data.get("movieId")
#         if not user_id or not movie_id:
#             return jsonify({"error": "Missing userId or movieId"}), 400

#         predicted_rating, approach = hybrid_predict(user_id, movie_id)
#         print(f"[DEBUG] final predicted_rating={predicted_rating}, approach={approach}")
#         return jsonify({
#             "predictedRating": round(predicted_rating, 2),
#             "approachUsed": approach
#         }), 200

#     except Exception as e:
#         print(f"Error in /api/predict-rating: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
from .hybrid_predict import hybrid_predict

app = Flask(__name__)

# Define allowed origins
ALLOWED_ORIGINS = [
    "https://popcornpair-6403c0694200.herokuapp.com",
    "http://localhost:8080"
]

# Enable CORS for specified origins (for routes that Flask-CORS will handle automatically)
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

@app.after_request
def add_cors_headers(response):
    # Get the Origin header from the incoming request
    origin = request.headers.get("Origin")
    # If the request's origin is in our allowed list, use it.
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
    else:
        # Otherwise, default to the first allowed origin.
        response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGINS[0]
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,PUT,POST,DELETE,OPTIONS"
    return response

# @app.route("/api/predict-rating", methods=["POST", "OPTIONS"])
# def predict_rating_endpoint():
#     if request.method == "OPTIONS":
#         # Flask-CORS and our after_request handler should now handle OPTIONS automatically.
#         return jsonify({}), 200

#     try:
#         data = request.json
#         user_id = data.get("userId")
#         movie_id = data.get("movieId")
#         if not user_id or not movie_id:
#             return jsonify({"error": "Missing userId or movieId"}), 400

#         predicted_rating, approach = hybrid_predict(user_id, movie_id)
#         print(f"[DEBUG] final predicted_rating={predicted_rating}, approach={approach}")

#         return jsonify({
#             "predictedRating": round(predicted_rating, 2),
#             "approachUsed": approach
#         }), 200

#     except Exception as e:
#         print(f"Error in /api/predict-rating: {str(e)}")
#         return jsonify({"error": str(e)}), 500

@app.route("/api/predict-rating", methods=["POST"])
def predict_rating_endpoint():
    try:
        data = request.json
        user_id = data.get("userId")
        movie_id = data.get("movieId")
        if not user_id or not movie_id:
            return jsonify({"error": "Missing userId or movieId"}), 400

        predicted_rating, approach = hybrid_predict(user_id, movie_id)
        print(f"[DEBUG] final predicted_rating={predicted_rating}, approach={approach}")

        return jsonify({
            "predictedRating": round(predicted_rating, 2),
            "approachUsed": approach
        }), 200

    except Exception as e:
        print(f"Error in /api/predict-rating: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
