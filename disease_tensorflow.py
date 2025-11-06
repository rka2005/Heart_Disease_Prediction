import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import os # <-- 1. Import os

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

label_encoders = {}
scaler = StandardScaler()
model = None
feature_order = []

# Define file paths
csv_data = "data/heart_disease.csv"
output_dir = "train" # <-- 2. Define your output folder

# Create the output directory if it doesn't exist (optional, but good practice)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

try:
    # --- 3. Fixed bug: Use the variable csv_data, not the string "csv_data" ---
    data = pd.read_csv(csv_data) 

    num_cols = data.select_dtypes(include=np.number).columns
    for col in num_cols:
        median_val = data[col].median()
        data[col] = data[col].fillna(median_val)
        
    cat_cols = data.select_dtypes(include='object').columns
    for col in cat_cols:
        mode_val = data[col].mode()[0]
        data[col] = data[col].fillna(mode_val)

    data['Stress Level'] = data['Stress Level'].map({'Low': 1, 'Medium': 2, 'High': 3})
    
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le
            if col == 'Heart Disease Status':
                print(f"Target mapping for 'Heart Disease Status': {dict(zip(le.classes_, le.transform(le.classes_)))}")

    X = data.drop('Heart Disease Status', axis=1)
    y = data['Heart Disease Status']
    
    feature_order = X.columns.tolist()
    
    print(f"\nTraining with {X.shape[1]} features: {feature_order}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    model = Sequential([
        tf.keras.Input(shape=(X_train_scaled.shape[1],)), # Use scaled data shape
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

    es = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

    print("\n--- Starting Model Training ---")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.15,
        callbacks=[es],
        class_weight=class_weight_dict,
        verbose=1 
    )
    print("--- Model Training Finished ---")

    y_prob = model.predict(X_test_scaled).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    print(f"\n--- Improved Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['No Disease (0)', 'Disease (1)']))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # --- 4. Modified save paths ---
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
        
        new_prob = model.predict(new_data_scaled).ravel()[0]
        new_pred = int(new_prob > 0.5)

        if new_pred == 1:
            print(f"\n==================================================================")
            print(f"⚠️ High risk: Likely heart disease detected. (Confidence: {new_prob * 100:.2f}%)")
            print(f"==================================================================")
        else:
            print(f"\n=================================================================")
            print(f"✅ Low risk: No heart disease detected. (Confidence: {(1 - new_prob) * 100:.2f}%)")
            print(f"=================================================================")
            
        plt.figure(figsize=(7, 5))
        
        labels = ['Risk of Disease (Class 1)', 'Confidence in No Disease (Class 0)']
        probabilities = [new_prob, 1 - new_prob]
        
        colors = ['#FFB4B4', '#4CAF50']
        if new_pred == 1:
            colors = ['#D32F2F', '#B4FFB4']

        bars = plt.bar(labels, probabilities, color=colors)
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval*100:.2f}%', ha='center', va='bottom')
            
        plt.ylabel('Probability')
        plt.title('Prediction Confidence for Your Input')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # --- 4. Modified save path ---
        pred_plot_path = os.path.join(output_dir, "prediction_confidence.png")
        plt.savefig(pred_plot_path)
        plt.close()
        
        print(f"\nA graph of this specific prediction has been saved to '{pred_plot_path}'")

    
    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")
        print("Please ensure you enter valid numerical values and correct options.")


except FileNotFoundError:
    # --- 3. Updated error message ---
    print(f"Error: The file '{csv_data}' was not found.") 
except Exception as e:
    print(f"An error occurred: {e}")