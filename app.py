from flask import Flask, request, jsonify
import pandas as pd
import joblib


# ---------------------------------------------------------
# Create Flask application
# ---------------------------------------------------------
app = Flask(__name__)


# ---------------------------------------------------------
# Load the trained end-to-end ML pipeline
# The pipeline contains:
#   1. Feature engineering
#   2. Data preprocessing / encoding
#   3. Trained Decision Tree classifier
# ---------------------------------------------------------
model_pipeline = joblib.load("model/churn_model.pkl")


# ---------------------------------------------------------
# Define the input fields required by the prediction API.
# These are the original customer attributes expected
# in the JSON request. Feature-engineered fields such as
# ServiceCount and IsNewCustomer are created automatically
# by the pipeline.
# ---------------------------------------------------------
REQUIRED_FIELDS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]


# ---------------------------------------------------------
# Prediction endpoint
# Accepts customer information as a JSON request and
# returns the predicted churn class and probability.
# ---------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict_churn():

    try:
        # Read the JSON request body.
        data = request.get_json()

        # Validate that the request contains data.
        if not data:
            return jsonify({
                "error": "Request body is empty"
            }), 400

        # Check whether any required customer fields are missing.
        missing_fields = [
            field
            for field in REQUIRED_FIELDS
            if field not in data
        ]

        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing_fields
            }), 400

        # Convert the JSON customer data into a one-row
        # Pandas DataFrame expected by the ML pipeline.
        customer_data = pd.DataFrame([data])

        # Generate the predicted churn class.
        prediction = model_pipeline.predict(customer_data)[0]

        # Generate prediction probabilities for both classes.
        probabilities = model_pipeline.predict_proba(customer_data)[0]

        # Retrieve the class labels used by the trained classifier.
        classes = model_pipeline.named_steps["classifier"].classes_

        # Find the probability corresponding to the predicted class.
        prediction_probability = probabilities[
            list(classes).index(prediction)
        ]

        # Return the prediction and probability as JSON.
        return jsonify({
            "churn_prediction": prediction,
            "churn_probability": round(
                float(prediction_probability), 4
            )
        })

    except Exception as e:
        # Handle invalid input or unexpected prediction errors
        # and return a meaningful HTTP 400 response.
        return jsonify({
            "error": "Invalid input or prediction error",
            "details": str(e)
        }), 400


# ---------------------------------------------------------
# Start the Flask development server.
# debug=False prevents Flask debug mode from being enabled
# in the submitted application.
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(
        debug=False,
        port=5000
    )