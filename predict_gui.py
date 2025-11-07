import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib
import os

# Load the trained model and scaler
model = joblib.load(os.path.join(os.path.dirname(__file__), "models", "ada_heart_model.pkl"))
scaler = joblib.load(os.path.join(os.path.dirname(__file__), "models", "heart_scaler_7param.pkl"))

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
root.title("Heart Disease Prediction")

# Dictionary to hold input widgets
entries = {}

# Create labels and inputs for each feature
for i, feat in enumerate(selected_features):
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
    tk.Label(root, text=label_text).grid(row=i, column=0, sticky='e', padx=5, pady=2)

    if feat == 'Gender':
        var = tk.StringVar(value='Male')
        tk.OptionMenu(root, var, 'Male', 'Female').grid(row=i, column=1, padx=5, pady=2)
        entries[feat] = var
    elif feat == 'Smoking':
        var = tk.StringVar(value='No')
        tk.OptionMenu(root, var, 'No', 'Yes').grid(row=i, column=1, padx=5, pady=2)
        entries[feat] = var
    elif feat == 'Stress Level':
        var = tk.StringVar(value='Medium')
        tk.OptionMenu(root, var, 'Low', 'Medium', 'High').grid(row=i, column=1, padx=5, pady=2)
        entries[feat] = var
    elif feat in ['Family Heart Disease', 'Diabetes']:
        var = tk.StringVar(value='No')
        tk.OptionMenu(root, var, 'No', 'Yes').grid(row=i, column=1, padx=5, pady=2)
        entries[feat] = var
    elif feat in ['Alcohol Consumption', 'Sugar Consumption']:
        var = tk.StringVar(value='Low')
        tk.OptionMenu(root, var, 'Low', 'Medium', 'High').grid(row=i, column=1, padx=5, pady=2)
        entries[feat] = var
    else:
        entry = tk.Entry(root)
        entry.grid(row=i, column=1, padx=5, pady=2)
        entries[feat] = entry

# BMI calculation fields
tk.Label(root, text="Weight (kg):").grid(row=len(selected_features), column=0, sticky='e', padx=5, pady=2)
weight_entry = tk.Entry(root)
weight_entry.grid(row=len(selected_features), column=1, padx=5, pady=2)

tk.Label(root, text="Height (feet):").grid(row=len(selected_features)+1, column=0, sticky='e', padx=5, pady=2)
feet_entry = tk.Entry(root)
feet_entry.grid(row=len(selected_features)+1, column=1, padx=5, pady=2)

tk.Label(root, text="Height (inches):").grid(row=len(selected_features)+2, column=0, sticky='e', padx=5, pady=2)
inches_entry = tk.Entry(root)
inches_entry.grid(row=len(selected_features)+2, column=1, padx=5, pady=2)

# Prediction function
def predict():
    try:
        data = {}
        for feat in selected_features:
            if isinstance(entries[feat], tk.StringVar):
                val = entries[feat].get()
                if feat == 'Gender':
                    data[feat] = 1 if val == 'Male' else 0
                elif feat in ['Smoking', 'Family Heart Disease', 'Diabetes']:
                    data[feat] = 1 if val == 'Yes' else 0
                elif feat == 'Stress Level':
                    data[feat] = {'Low': 1, 'Medium': 2, 'High': 3}[val]
                elif feat in ['Alcohol Consumption', 'Sugar Consumption']:
                    data[feat] = {'Low': 0, 'Medium': 1, 'High': 2}[val]
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

        if weight <= 0 or feet <= 0 or inches < 0:
            raise ValueError("Weight and height must be positive values")

        height_m = (feet * 0.3048) + (inches * 0.0254)
        if height_m <= 0:
            raise ValueError("Invalid height calculation")

        bmi = weight / (height_m ** 2)
        data['BMI'] = bmi

        df = pd.DataFrame([data], columns=feature_order)
        df_scaled = scaler.transform(df)
        probability = model.predict_proba(df_scaled)[0][1] * 100
        prediction = 1 if probability > 45 else 0

        if prediction == 1:
            msg = f"⚠️ High risk: Likely heart disease detected. (Confidence: {probability:.2f}%)"
        else:
            msg = f"✅ Low risk: No heart disease detected. (Confidence: {100 - probability:.2f}%)"

        messagebox.showinfo("Prediction Result", msg)
    except ValueError as e:
        messagebox.showerror("Error", f"Please enter valid numerical values for all fields.\n\nDetails: {str(e)}")
    except ZeroDivisionError:
        messagebox.showerror("Error", "Height cannot be zero. Please enter valid height.")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

# Predict button
tk.Button(root, text="Predict", command=predict).grid(row=len(selected_features)+3, column=1, pady=10)

# Run the GUI
root.mainloop()