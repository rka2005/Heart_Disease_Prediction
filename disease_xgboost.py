import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os
import joblib
import numpy as np
import time

warnings.filterwarnings('ignore')

start_time = time.time()

print("\n" + "="*80)
print("🏥 HEART DISEASE PREDICTION - XGBOOST FAST MODEL 🏥")
print("="*80)

# ============================================================================
# 1️⃣ LOAD & PREPROCESS DATA
# ============================================================================
print("\n[1/4] Loading and preprocessing data...")
data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "heart_disease.csv"))

# Handle missing values
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].fillna(data[col].mode()[0])
    else:
        data[col] = data[col].fillna(data[col].mean())

# ENCODE TARGET VARIABLE
data['Heart Disease Status'] = (data['Heart Disease Status'] == 'Yes').astype(int)

# Encode categorical features
label_encoders = {}
for col in data.columns:
    if data[col].dtype == 'object':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

# ============================================================================
# 2️⃣ SELECT FEATURES
# ============================================================================
print("[2/4] Selecting features...")
selected_features = [
    'Age', 'Cholesterol Level', 'Blood Pressure', 'CRP Level', 'Smoking', 'Diabetes', 'BMI'
]

X = data[selected_features]
y = data['Heart Disease Status']

print(f"     Dataset: {len(X)} samples, {len(selected_features)} features")
print(f"     Class Distribution - No Disease: {(y==0).sum()}, Disease: {(y==1).sum()}")

# ============================================================================
# 3️⃣ TRAIN-TEST SPLIT
# ============================================================================
print("[3/4] Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"     Training: {len(X_train)} samples, Testing: {len(X_test)} samples")

# ============================================================================
# 4️⃣ TRAIN XGBOOST MODEL
# ============================================================================
print("[4/4] Training XGBoost model...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    verbosity=0,
    tree_method='hist',  # Faster histogram-based training
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
elapsed = time.time() - start_time
print(f"     Training completed in {elapsed:.2f} seconds")

# ============================================================================
# EVALUATE MODEL
# ============================================================================
print("\n" + "="*80)
print("📊 MODEL EVALUATION")
print("="*80)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
try:
    auc = roc_auc_score(y_test, y_pred_proba)
except:
    auc = 0

print(f"\n✅ Accuracy:  {accuracy*100:.2f}%")
print(f"✅ F1-Score:  {f1:.4f}")
print(f"✅ ROC-AUC:   {auc:.4f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Disease (0)', 'Disease (1)']))

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
print(f"   True Negatives:  {tn:4d} | False Positives: {fp:4d}")
print(f"   False Negatives: {fn:4d} | True Positives:  {tp:4d}")
print(f"   Sensitivity (Recall):     {sensitivity*100:.2f}%")
print(f"   Specificity (TNR):        {specificity*100:.2f}%")

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n[Saving models...]")
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

joblib.dump(model, os.path.join(models_dir, "heart_disease_model.pkl"))
joblib.dump(scaler, os.path.join(models_dir, "heart_disease_scaler.pkl"))

print(f"   ✓ XGBoost model saved")
print(f"   ✓ Scaler saved")

# ============================================================================
# PLOT FEATURE IMPORTANCES
# ============================================================================
print("\n[Creating visualizations...]")

try:
    # Get feature importances
    importances = model.feature_importances_
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    indices = np.argsort(importances)[::-1]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_features)))
    bars = ax.barh(np.array(selected_features)[indices], importances[indices], color=colors)
    
    ax.set_xlabel("Feature Importance Score", fontsize=12, fontweight='bold')
    ax.set_ylabel("Features", fontsize=12, fontweight='bold')
    ax.set_title("Heart Disease Prediction - XGBoost Feature Importances", 
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    plot_path = os.path.join(models_dir, "heart_disease_feature_importances.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Feature importance plot saved")
    
except Exception as e:
    print(f"   ⚠ Error creating plot: {str(e)}")

# ============================================================================
# TRAINING SUMMARY
# ============================================================================
total_time = time.time() - start_time

print("\n" + "="*80)
print("✨ TRAINING COMPLETE ✨")
print("="*80)
print(f"\n⏱️  Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"\n🎯 Final Model Performance:")
print(f"   • Accuracy:  {accuracy*100:.2f}%")
print(f"   • F1-Score:  {f1:.4f}")
print(f"   • ROC-AUC:   {auc:.4f}")
print(f"\n📁 Models saved to: {models_dir}")
print(f"\n📚 Model Info:")
print(f"   • Model Type: XGBoost (eXtreme Gradient Boosting)")
print(f"   • Estimators: 200")
print(f"   • Tree Method: Histogram-based (FAST)")
print(f"   • Features: {len(selected_features)} selected")
print(f"   • Training Speed: ⭐⭐⭐ (Fastest)")
print(f"\n💡 Ready for predictions! Use predict_gui.py")
print("="*80 + "\n")
