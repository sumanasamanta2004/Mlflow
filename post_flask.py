from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn

# Initialize Flask app
app = Flask(__name__)

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:8081")

# Correct model URI — use a valid stage name or version
model_uri = "models:/RandomForest_Classifier@champion"
model = mlflow.sklearn.load_model(model_uri)

@app.route('/')                     
def home():
    return "🎯 RandomForest MLflow Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json(force=True)
        features = input_data.get("features", None)

        if features is None or len(features) != 4:
            return jsonify({"error": "Invalid input. Provide 4 numeric values in 'features' list."}), 400

        # Predict
        prediction = model.predict([features])
        result = prediction[0]  # Might be a string like 'setosa'

        return jsonify({
            "input": features,
            "prediction": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
