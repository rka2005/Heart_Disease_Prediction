import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib
import os

# =============================================================================
# HEART DISEASE PREDICTION - GUI INTERFACE
# =============================================================================
# This GUI loads the trained XGBoost model and scaler to make real-time
# predictions based on user input of 7 health parameters.
# 
# Features:
# - Age, Cholesterol Level, Blood Pressure, CRP Level, Smoking, Diabetes, BMI
# - BMI is calculated from weight and height
# - Predictions show disease probability and confidence level
# =============================================================================

# Load the trained model and scaler
model = joblib.load(os.path.join(os.path.dirname(__file__), "models", "heart_disease_model.pkl"))
scaler = joblib.load(os.path.join(os.path.dirname(__file__), "models", "heart_disease_scaler.pkl"))

# Selected features (top important ones for simplicity)
selected_features = [
    'Age', 'Cholesterol Level', 'Blood Pressure', 'CRP Level', 'Smoking', 'Diabetes'
]

# Feature order must match training data (same as scaler expects)
feature_order = [
    'Age', 'Cholesterol Level', 'Blood Pressure', 'CRP Level', 'Smoking', 'Diabetes', 'BMI'
]

# Create the main window
root = tk.Tk()
root.title("🏥 Heart Disease Prediction System - XGBoost Model")
root.geometry("450x550")
root.resizable(False, False)

# Add title label
title_label = tk.Label(root, text="❤️ Heart Disease Risk Prediction", 
                       font=("Arial", 14, "bold"), fg="darkred")
title_label.grid(row=0, column=0, columnspan=2, pady=10)

# Dictionary to hold input widgets
entries = {}

# Create labels and inputs for each feature
for i, feat in enumerate(selected_features):
    row = i + 1  # Start from row 1 (row 0 has title)
    
    if feat == 'Age':
        label_text = "Age (years):"
    elif feat == 'BMI':
        label_text = "BMI (Body Mass Index):"
    elif feat == 'Cholesterol Level':
        label_text = "Cholesterol Level (mg/dL):"
    elif feat == 'Blood Pressure':
        label_text = "Blood Pressure (mmHg):"
    elif feat == 'CRP Level':
        label_text = "CRP Level (C-Reactive Protein, mg/L):"
    else:
        label_text = feat + ":"
    tk.Label(root, text=label_text).grid(row=row, column=0, sticky='e', padx=5, pady=2)

    if feat == 'Smoking':
        var = tk.StringVar(value='No')
        tk.OptionMenu(root, var, 'No', 'Yes').grid(row=row, column=1, padx=5, pady=2)
        entries[feat] = var
    elif feat == 'Diabetes':
        var = tk.StringVar(value='No')
        tk.OptionMenu(root, var, 'No', 'Yes').grid(row=row, column=1, padx=5, pady=2)
        entries[feat] = var
    else:
        entry = tk.Entry(root)
        entry.grid(row=row, column=1, padx=5, pady=2)
        entries[feat] = entry

# BMI calculation fields
num_features = len(selected_features)
tk.Label(root, text="Weight (kg):").grid(row=num_features+1, column=0, sticky='e', padx=5, pady=2)
weight_entry = tk.Entry(root)
weight_entry.grid(row=num_features+1, column=1, padx=5, pady=2)

tk.Label(root, text="Height (feet):").grid(row=num_features+2, column=0, sticky='e', padx=5, pady=2)
feet_entry = tk.Entry(root)
feet_entry.grid(row=num_features+2, column=1, padx=5, pady=2)

tk.Label(root, text="Height (inches):").grid(row=num_features+3, column=0, sticky='e', padx=5, pady=2)
inches_entry = tk.Entry(root)
inches_entry.grid(row=num_features+3, column=1, padx=5, pady=2)

# Prediction function
def predict():
    try:
        data = {}
        for feat in selected_features:
            if isinstance(entries[feat], tk.StringVar):
                val = entries[feat].get()
                # Convert Yes/No to 1/0
                if feat in ['Smoking', 'Diabetes']:
                    data[feat] = 1 if val == 'Yes' else 0
                else:
                    data[feat] = float(val)
            else:
                # Get the value and check if it's empty or invalid
                val = entries[feat].get().strip()
                if not val:
                    raise ValueError(f"{feat} cannot be empty")
                data[feat] = float(val)

        # Calculate BMI
        weight_val = weight_entry.get().strip()
        feet_val = feet_entry.get().strip()
        inches_val = inches_entry.get().strip()

        if not weight_val or not feet_val or not inches_val:
            raise ValueError("Weight, feet, and inches cannot be empty")

        weight = float(weight_val)
        feet = float(feet_val)
        inches = float(inches_val)

        if weight <= 0 or feet < 0 or inches < 0:
            raise ValueError("Weight and height must be positive values")

        # Convert feet and inches to meters
        height_m = (feet * 0.3048) + (inches * 0.0254)
        if height_m <= 0:
            raise ValueError("Invalid height calculation")

        # Calculate BMI
        bmi = weight / (height_m ** 2)
        data['BMI'] = bmi

        # Create DataFrame matching feature order
        df = pd.DataFrame([data], columns=feature_order)
        
        # Scale the data
        df_scaled = scaler.transform(df)
        
        # Get prediction probability
        probability = model.predict_proba(df_scaled)[0][1]
        risk_percentage = probability * 100
        confidence_percentage = (1 - probability) * 100 if probability <= 0.5 else probability * 100

        # Determine risk level
        if probability > 0.5:
            msg = f"⚠️ HIGH RISK\n\nHeart Disease Probability: {risk_percentage:.2f}%\nConfidence: {risk_percentage:.2f}%"
        else:
            msg = f"✅ LOW RISK\n\nHeart Disease Probability: {risk_percentage:.2f}%\nConfidence: {confidence_percentage:.2f}%"

        messagebox.showinfo("Prediction Result", msg)
    except ValueError as e:
        messagebox.showerror("Input Error", f"Please enter valid values.\n\nDetails: {str(e)}")
    except ZeroDivisionError:
        messagebox.showerror("Calculation Error", "Height cannot be zero. Please enter valid height.")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred:\n{str(e)}")

# Predict button
predict_button = tk.Button(root, text="🔍 Predict", command=predict, 
                           bg="darkgreen", fg="white", font=("Arial", 12, "bold"),
                           padx=20, pady=10)
predict_button.grid(row=num_features+4, column=0, columnspan=2, pady=15)

# Info label
info_label = tk.Label(root, text="Model: XGBoost | Accuracy: 78.65% | Training: 1.02s", 
                      font=("Arial", 8), fg="gray")
info_label.grid(row=num_features+5, column=0, columnspan=2, pady=5)

# Run the GUI
root.mainloop()