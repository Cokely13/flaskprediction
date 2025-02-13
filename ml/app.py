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
