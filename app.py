from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("gradient_boosting_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define the expected categorical columns for encoding
categorical_cols = ['Brand', 'Size', 'Material', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

# Sample categories from training data (modify based on actual categories)
category_values = {
    "Brand": ["Jansport", "Nike", "Adidas", "Puma", "Under Armour"],
    "Size": ["Small", "Medium", "Large"],
    "Material": ["Canvas", "Leather", "Nylon", "Polyester"],
    "Laptop Compartment": ["Yes", "No"],
    "Waterproof": ["Yes", "No"],
    "Style": ["Backpack", "Messenger", "Tote"],
    "Color": ["Black", "Blue", "Green", "Gray", "Pink", "Red"]
}

@app.route("/")
def home():
    return render_template("index.html", categories=category_values)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input from the form
        user_input = {key: request.form[key] for key in request.form}

        # Convert input into a DataFrame
        input_df = pd.DataFrame([user_input])

        # Convert numerical values properly
        input_df["Compartments"] = float(input_df["Compartments"].iloc[0])
        input_df["Weight Capacity (kg)"] = float(input_df["Weight Capacity (kg)"].iloc[0])

        # One-hot encode categorical values
        input_df = pd.get_dummies(input_df)

        # Align with training data: Ensure all expected columns exist
        missing_cols = set(scaler.feature_names_in_) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0  # Add missing columns as 0

        # Reorder columns to match the model training order
        input_df = input_df[scaler.feature_names_in_]

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict price
        predicted_price = model.predict(input_scaled)[0]

        return render_template("index.html", categories=category_values, prediction=f"Estimated Price: ${predicted_price:.2f}")

    except Exception as e:
        return render_template("index.html", categories=category_values, error=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
