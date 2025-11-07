"""
Heart Disease Prediction System - TensorFlow Implementation
Uses deep neural networks for medical risk assessment
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import warnings
import os
import pickle

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ========================= GLOBAL VARIABLES =========================
label_encoders = {}
scaler = StandardScaler()
model = None
feature_order = []

# ========================= FILE PATHS CONFIGURATION =========================
script_dir = os.path.dirname(__file__)
csv_data = os.path.abspath(os.path.join(script_dir, 'data', 'heart_disease.csv'))

# Fallback paths
if not os.path.exists(csv_data):
    alt = os.path.abspath(os.path.join(script_dir, '..', 'XGBoost', 'data', 'heart_disease.csv'))
    if os.path.exists(alt):
        csv_data = alt
    else:
        alt2 = os.path.abspath(os.path.join(script_dir, '..', 'data', 'heart_disease.csv'))
        if os.path.exists(alt2):
            csv_data = alt2

print(f"✓ Using CSV path: {csv_data}")
output_dir = os.path.join(script_dir, "train")

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"✓ Created directory: {output_dir}")

try:
    # ========================= DATA LOADING =========================
    print("\n" + "="*70)
    print("PHASE 1: DATA LOADING & PREPROCESSING")
    print("="*70)
    
    print("\n[1/5] Loading dataset...")
    data = pd.read_csv(csv_data)
    print(f"✓ Dataset loaded. Shape: {data.shape}")

    # ========================= MISSING VALUE HANDLING =========================
    print("[2/5] Handling missing values...")
    num_cols = data.select_dtypes(include=np.number).columns
    for col in num_cols:
        median_val = data[col].median()
        data[col] = data[col].fillna(median_val)
        
    cat_cols = data.select_dtypes(include='object').columns
    for col in cat_cols:
        mode_val = data[col].mode()[0] if len(data[col].mode()) > 0 else "Unknown"
        data[col] = data[col].fillna(mode_val)
    
    print("✓ Missing values handled")

    # ========================= CATEGORICAL ENCODING =========================
    print("[3/5] Encoding categorical variables...")
    
    # Handle Stress Level with custom mapping
    if 'Stress Level' in data.columns:
        stress_le = LabelEncoder()
        stress_le.fit(['Low', 'Medium', 'High'])
        data['Stress Level'] = stress_le.transform(data['Stress Level'].astype(str))
        label_encoders['Stress Level'] = stress_le
    
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le
            if col == 'Heart Disease Status':
                print(f"  Target mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    print(f"✓ Encoded {len(label_encoders)} categorical variables")

    # ========================= FEATURE & TARGET SEPARATION =========================
    print("[4/5] Separating features and target...")
    X = data.drop('Heart Disease Status', axis=1)
    y = data['Heart Disease Status']
    
    feature_order = X.columns.tolist()
    
    print(f"✓ Features: {X.shape[1]}")
    print(f"  Samples: {len(X)}")
    print(f"  Class distribution: {dict(y.value_counts())}")

    # ========================= TRAIN-TEST SPLIT =========================
    print("[5/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ========================= FEATURE SCALING =========================
    print("\n" + "="*70)
    print("PHASE 2: MODEL TRAINING")
    print("="*70)
    
    print("\n[1/4] Scaling features...")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"✓ Features scaled (Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape})")

    # ========================= CLASS WEIGHTS =========================
    print("[2/4] Computing class weights for imbalanced data...")
    class_weights = class_weight.compute_class_weight(
        'balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"✓ Class weights: {class_weight_dict}")

    # ========================= MODEL ARCHITECTURE =========================
    print("[3/4] Building TensorFlow neural network...")
    model = Sequential([
        tf.keras.Input(shape=(X_train_scaled.shape[1],)),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(), 
                 tf.keras.metrics.Recall(), 
                 tf.keras.metrics.AUC()]
    )
    print("✓ Model architecture created")

    # ========================= MODEL TRAINING =========================
    print("[4/4] Training model...")
    es = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0)

    print("\n--- Training Progress ---")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=300,
        batch_size=64,
        validation_split=0.15,
        callbacks=[es, reduce_lr],
        class_weight=class_weight_dict,
        verbose=1
    )
    print("✓ Training completed")

    # ========================= MODEL EVALUATION =========================
    print("\n" + "="*70)
    print("PHASE 3: MODEL EVALUATION")
    print("="*70)
    
    print("\n[1/3] Computing predictions...")
    y_prob = model.predict(X_test_scaled, verbose=0).ravel()
    y_pred = (y_prob > 0.5).astype(int)
    
    train_prob = model.predict(X_train_scaled, verbose=0).ravel()
    train_pred = (train_prob > 0.5).astype(int)
    print("✓ Predictions computed")

    print("\n[2/3] Performance Metrics:")
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)
    train_auc = roc_auc_score(y_train, train_prob)
    test_auc = roc_auc_score(y_test, y_prob)
    
    print(f"  Train Accuracy: {train_accuracy*100:.2f}%")
    print(f"  Test Accuracy:  {test_accuracy*100:.2f}%")
    print(f"  Train AUC-ROC:  {train_auc:.4f}")
    print(f"  Test AUC-ROC:   {test_auc:.4f}")

    print("\n" + "-"*70)
    print("Classification Report:")
    print("-"*70)
    print(classification_report(y_test, y_pred, target_names=['No Disease (0)', 'Disease (1)']))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n[3/3] Saving model artifacts...")
    
    # Save model
    model_save_path = os.path.join(output_dir, "tf_heart_model.keras")
    model.save(model_save_path)
    print(f"✓ Model saved: {model_save_path}")
    
    # Save scaler
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved: {scaler_path}")
    
    # Save encoders
    encoders_path = os.path.join(output_dir, "label_encoders.pkl")
    with open(encoders_path, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"✓ Encoders saved: {encoders_path}")
    
    # Save feature order
    feature_path = os.path.join(output_dir, "feature_order.pkl")
    with open(feature_path, 'wb') as f:
        pickle.dump(feature_order, f)
    print(f"✓ Feature order saved: {feature_path}")

    # ========================= VISUALIZATIONS =========================
    print("\n" + "="*70)
    print("PHASE 4: GENERATING VISUALIZATIONS")
    print("="*70)
    
    print("\n[1/5] Training history...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    plt.title('Accuracy', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
    plt.title('Loss', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['auc'], label='Train', linewidth=2)
    plt.plot(history.history['val_auc'], label='Validation', linewidth=2)
    plt.title('AUC-ROC', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    training_plot_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(training_plot_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {training_plot_path}")
    
    print("[2/5] ROC Curve...")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {test_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('ROC Curve', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {roc_path}")
    
    print("[3/5] Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {cm_path}")
    
    print("[4/5] Prediction Distribution...")
    plt.figure(figsize=(10, 6))
    plt.hist(y_prob[y_test == 0], bins=30, alpha=0.6, label='No Disease (Actual)', color='green')
    plt.hist(y_prob[y_test == 1], bins=30, alpha=0.6, label='Disease (Actual)', color='red')
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
    plt.xlabel('Prediction Probability', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.title('Distribution of Prediction Probabilities', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    dist_path = os.path.join(output_dir, "prediction_distribution.png")
    plt.savefig(dist_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {dist_path}")
    
    print("[5/5] Performance Summary...")
    fig, ax = plt.subplots(figsize=(10, 6))
    from sklearn.metrics import precision_score, recall_score
    metrics_names = ['Accuracy', 'AUC-ROC', 'Precision', 'Recall']
    metrics_values = [
        test_accuracy,
        test_auc,
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred)
    ]
    bars = ax.bar(metrics_names, metrics_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    for bar, val in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Model Performance Metrics', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    perf_path = os.path.join(output_dir, "performance_summary.png")
    plt.savefig(perf_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {perf_path}")

    # ========================= INTERACTIVE PREDICTION =========================
    print("\n" + "="*70)
    print("PHASE 5: INTERACTIVE PREDICTIONS")
    print("="*70)
    
    while True:
        print("\n[1] Make a prediction")
        print("[2] Test on random sample")
        print("[3] Exit")
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            try:
                input_dict = {}
                print("\n" + "-"*70)
                for col in feature_order:
                    if col in label_encoders:
                        le = label_encoders[col]
                        print(f"\n{col}: {', '.join(le.classes_)}")
                        val_str = input(f"Enter {col}: ").strip()
                        if val_str not in le.classes_:
                            val_str = le.classes_[0]
                        input_dict[col] = le.transform([val_str])[0]
                    else:
                        val = float(input(f"Enter {col}: "))
                        input_dict[col] = val
                
                new_data = pd.DataFrame([input_dict])[feature_order]
                new_scaled = scaler.transform(new_data)
                prob = model.predict(new_scaled, verbose=0)[0][0]
                pred = 1 if prob > 0.5 else 0
                
                print("\n" + "="*70)
                if pred == 1:
                    print(f"⚠️ HIGH RISK: Heart disease likely (Confidence: {prob*100:.1f}%)")
                else:
                    print(f"✅ LOW RISK: No heart disease (Confidence: {(1-prob)*100:.1f}%)")
                print("="*70)
                
            except Exception as e:
                print(f"✗ Error: {e}")
        
        elif choice == '2':
            sample = data.sample(1).iloc[0]
            sample_dict = {col: sample[col] for col in feature_order}
            new_data = pd.DataFrame([sample_dict])[feature_order]
            new_scaled = scaler.transform(new_data)
            prob = model.predict(new_scaled, verbose=0)[0][0]
            pred = 1 if prob > 0.5 else 0
            actual = sample['Heart Disease Status']
            
            print(f"\nPrediction: {'Disease' if pred == 1 else 'No Disease'} ({max(prob, 1-prob)*100:.1f}%)")
            print(f"Actual: {'Disease' if actual == 1 else 'No Disease'}")
            print(f"Status: {'✓ CORRECT' if pred == actual else '✗ INCORRECT'}")
        
        elif choice == '3':
            print("\nExiting system...")
            break
    
    print("\n" + "="*70)
    print("✓ SYSTEM COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nAll files saved in: {output_dir}")

except Exception as e:
    print("\n" + "="*70)
    print("✗ ERROR OCCURRED!")
    print("="*70)
    print(f"Error: {str(e)}")
    print("\nTroubleshooting tips:")
    print("1. Ensure heart_disease.csv exists in the data/ folder")
    print("2. Check that all required packages are installed:")
    print("   pip install tensorflow scikit-learn pandas numpy matplotlib seaborn")
    print("3. Verify file permissions")
    print("="*70)