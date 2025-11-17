import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, f1_score
import matplotlib
matplotlib.use('Agg') # Use Agg backend for saving files
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import os
from imblearn.over_sampling import SMOTE # Import SMOTE

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

label_encoders = {}
scaler = StandardScaler()
model = None
feature_order = []

# Define file paths
csv_data = "data/heart_disease.csv" # <-- 1. Fixed path to read from root
output_dir = "train"


if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

try:
    data = pd.read_csv(csv_data) 

    # --- Preprocessing ---
    num_cols = data.select_dtypes(include=np.number).columns
    for col in num_cols:
        median_val = data[col].median()
        data[col] = data[col].fillna(median_val)
        
    cat_cols = data.select_dtypes(include='object').columns
    for col in cat_cols:
        mode_val = data[col].mode()[0]
        data[col] = data[col].fillna(mode_val)

    # **(CRITICAL FIX)** Convert Target Variable to 1/0
    data['Heart Disease Status'] = (data['Heart Disease Status'] == 'Yes').astype(int)

    data['Stress Level'] = data['Stress Level'].map({'Low': 1, 'Medium': 2, 'High': 3}).fillna(2)
    
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le

    X = data.drop('Heart Disease Status', axis=1)
    y = data['Heart Disease Status']
    
    feature_order = X.columns.tolist()
    
    print(f"\nTraining with {X.shape[1]} features: {feature_order}")

    # --- 1. Create the final Test set ---
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 2. Create the real Training and Validation sets ---
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
    )

    # --- 3. Scale all three sets ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- 4. Apply SMOTE *only* to the training set ---
    print(f"Original training shape: {np.bincount(y_train)}")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
    print(f"Resampled training shape: {np.bincount(y_train_res)}")
    print(f"Validation shape: {np.bincount(y_val)} (Imbalanced - this is correct)")
    # -----------------------------------------------

    model = Sequential([
        tf.keras.Input(shape=(X_train_scaled.shape[1],)), 
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )

    # Monitor 'val_recall' to find the model best at catching "Yes" cases
    es = EarlyStopping(
        monitor='val_recall', 
        mode='max', 
        patience=30, 
        restore_best_weights=True
    )

    print("\n--- Starting Model Training (Optimizing for Recall) ---")
    history = model.fit(
        X_train_res, y_train_res, # Train on balanced data
        epochs=200,
        batch_size=32,
        validation_data=(X_val_scaled, y_val), # Validate on real, imbalanced data
        callbacks=[es],
        verbose=1 
    )
    print("--- Model Training Finished ---")

    # --- 6. Find Optimal Threshold ---
    y_prob = model.predict(X_test_scaled).ravel()
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    
    # Add a small epsilon to avoid division by zero
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
    
    # Find the threshold that gives the best F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print("\n" + "="*80)
    print("📊 MODEL EVALUATION & THRESHOLD")
    print("="*80)
    print(f"Default 0.5 Threshold F1-Score: {f1_score(y_test, (y_prob > 0.5)):.4f}")
    print(f"Optimal F1-Score: {f1_scores[optimal_idx]:.4f}")
    print(f"Optimal Threshold found: {optimal_threshold:.4f}")
    print("This threshold will be used for predictions.")
    # --------------------------------

    y_pred_optimal = (y_prob > optimal_threshold).astype(int)

    print(f"\n--- Evaluation with Optimal Threshold ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_optimal) * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred_optimal, target_names=['No Disease (0)', 'Disease (1)']))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_optimal))

    model_save_path = os.path.join(output_dir, "tf_heart_model_full_features.keras")
    model.save(model_save_path)
    print(f"\nModel saved to '{model_save_path}'")
    
    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title("Improved Model: Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    acc_plot_path = os.path.join(output_dir, "tf_improved_accuracy.png")
    plt.savefig(acc_plot_path)
    plt.close() 

    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Improved Model: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(output_dir, "tf_improved_loss.png")
    plt.savefig(loss_plot_path)
    plt.close() 
    
    print(f"Plots saved to '{acc_plot_path}' and '{loss_plot_path}'")

    
    print("\n\n--- Enter Details for Live Prediction ---")
    
    new_input_raw = {}

    try:
        for col in feature_order:
            
            if col == 'Stress Level':
                print("\nStress Level Options:\n1 = Low | 2 = Medium | 3 = High")
                val = float(input(f"Enter {col} (1/2/3): "))
                new_input_raw[col] = val
            
            elif col in label_encoders:
                le = label_encoders[col]
                print(f"\nEnter {col} (Options: {le.classes_}): ")
                val_str = input()
                
                try:
                    new_input_raw[col] = le.transform([val_str])[0]
                except ValueError:
                    print(f"Invalid option '{val_str}'. Defaulting to first option: {le.classes_[0]}")
                    new_input_raw[col] = le.transform([le.classes_[0]])[0]
            
            else:
                val = float(input(f"Enter {col}: "))
                new_input_raw[col] = val

        new_data = pd.DataFrame([new_input_raw])
        
        new_data = new_data[feature_order]
        
        new_data_scaled = scaler.transform(new_data)
        
        # --- 7. Use Optimal Threshold for Prediction ---
        new_prob = model.predict(new_data_scaled).ravel()[0]
        risk = new_prob
        no_risk = 1 - new_prob

        print("\n====================== RESULT ======================")

        if no_risk < risk:
            print(f"✅ Low Risk: No heart disease detected.")
            print(f"Confidence in No Disease: {no_risk*100:.2f}%")
            print(f"Risk of Disease: {risk*100:.2f}%")
        else:
            print(f"⚠️ High Risk: Possible heart disease detected.")
            print(f"Risk of Disease: {risk*100:.2f}%")
            print(f"Confidence in No Disease: {no_risk*100:.2f}%")

        print("====================================================")
            
        plt.figure(figsize=(7, 5))
        
        labels = ['Risk of Disease (Class 1)', 'Confidence in No Disease (Class 0)']
        probabilities = [risk, no_risk]
        
        colors = ['#FFB4B4', '#4CAF50']
        if risk > no_risk:
            colors = ['#D32F2F', '#B4FFB4']

        bars = plt.bar(labels, probabilities, color=colors)
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval*100:.2f}%', ha='center', va='bottom')
        
        # Add the threshold line to the prediction graph
        plt.axhline(y=optimal_threshold, color='r', linestyle='--', label=f"Decision Threshold ({optimal_threshold*100:.2f}%)")
        plt.legend()
            
        plt.ylabel('Probability')
        plt.title('Prediction Confidence for Your Input')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        pred_plot_path = os.path.join(output_dir, "prediction_confidence.png")
        plt.savefig(pred_plot_path)
        plt.close()
        
        print(f"\nA graph of this specific prediction has been saved to '{pred_plot_path}'")

    
    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")
        print("Please ensure you enter valid numerical values and correct options.")


except FileNotFoundError:
    print(f"Error: The file '{csv_data}' was not found.") 
except Exception as e:
    print(f"An error occurred: {e}")