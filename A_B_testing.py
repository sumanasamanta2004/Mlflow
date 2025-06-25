from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn
import random

# Initialize Flask app
app = Flask(__name__)

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:8081")

# Pre-load both models from their respective registry stages
champion_uri = "models:/RandomForest_Classifier@champion"
challenger_uri = "models:/RandomForest_Classifier@challenger"

champion_model = mlflow.sklearn.load_model(champion_uri)
challenger_model = mlflow.sklearn.load_model(challenger_uri)

@app.route('/')
def home():
    return "🎯 RandomForest MLflow Model API (Champion vs Challenger) is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get input JSON
        input_data = request.get_json(force=True)
        features = input_data.get("features", None)

        # 2. Validate input
        if features is None or len(features) != 4:
            return jsonify({"error": "Invalid input. Provide 4 numeric values in 'features' list."}), 400

        # 3. Generate random number and decide model
        random_value = random.random()
        if random_value > 0.5:
            model_used = "challenger"
            model = challenger_model
        else:
            model_used = "champion"
            model = champion_model

        # 4. Make prediction
        prediction = model.predict([features])
        result = prediction[0]

        # 5. Return result with model used and random value
        return jsonify({
            "input": features,
            "prediction": result,
            "model_used": model_used,
            "random_value": round(random_value, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
