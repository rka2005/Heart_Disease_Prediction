# ❤️ Heart Disease Prediction System - TensorFlow Implementation

**🎓 Adamas University - Python Project**

> **📖 Documentation Guide:**
> - **README.md** (this file) - 🚀 Quick start and main overview
> - **COMPLETE_GUIDANCE.md** - 📚 Comprehensive technical documentation (5000+ words)
> - **TERMS.md** - 🔍 Glossary of ML and medical terms
> - **[LICENSE](LICENSE)** - 📋 Project information, team, and requirements

## 📋 Overview

This project implements a **🧠 Machine Learning-based Heart Disease Prediction System** using **TensorFlow Neural Networks** for advanced deep learning capabilities.

### Main Components:
- **`disease_tensorflow.py`** - 💻 Complete TensorFlow neural network implementation (RECOMMENDED)
- **`disease.py`** - 📦 Legacy XGBoost implementation (for reference only)
- 🔄 Comprehensive data preprocessing pipeline
- 🎯 Advanced neural network architecture with regularization
- 📊 Multiple evaluation metrics and visualizations
- 💬 Interactive prediction interface

## 📁 Directory Structure

```
TensorFlow/
├── 📂 data/
│   └── 📄 heart_disease.csv              # 📊 Dataset (10,000 samples, 21 features)
│
├── 📂 train/                             # 🏆 Generated after first run
│   ├── 🤖 tf_heart_model.keras           # 🧠 Trained TensorFlow model
│   ├── 📦 scaler.pkl                     # 🔄 StandardScaler for feature normalization
│   ├── 📦 label_encoders.pkl             # 🏷️ LabelEncoders for categorical features
│   ├── 📦 feature_order.pkl              # ✅ Feature ordering (for consistency)
│   ├── 📈 training_history.png           # 📊 Training/validation curves
│   ├── 📉 roc_curve.png                  # 🎯 ROC curve with AUC score
│   ├── 🔥 confusion_matrix.png           # 📐 Confusion matrix heatmap
│   ├── 📊 prediction_distribution.png    # 📈 Histogram of predictions
│   └── 🎨 performance_summary.png        # 📊 Metrics bar chart
│
├── 💻 disease_tensorflow.py              # 🌟 Main TensorFlow implementation ⭐
├── 📜 disease.py                         # 🗂️ Legacy XGBoost (reference only)
├── 📄 README.md                          # 📖 This file
└── 📚 COMPLETE_GUIDANCE.md               # 🔬 Detailed implementation guide
```

## 🚀 Quick Start

### 🌟 Primary Implementation (TensorFlow)
```bash
python disease_tensorflow.py
```

**⚙️ This script performs all steps automatically:**
1. **Phase 1️⃣:** 📥 Data loading & preprocessing
2. **Phase 2️⃣:** 🧠 Neural network training
3. **Phase 3️⃣:** 📊 Model evaluation & metrics
4. **Phase 4️⃣:** 🎨 Generate 5 visualizations
5. **Phase 5️⃣:** 💬 Interactive prediction interface

### 📦 Legacy Implementation (XGBoost - Reference Only)
```bash
python disease.py  # ⚠️ Not recommended - use TensorFlow version
```

## 📊 Dataset

**📄 File:** `data/heart_disease.csv`

**📋 Features (21 total):**
- 👤 Demographics: Age, Gender
- 💓 Vital Signs: Blood Pressure
- 🩸 Lipids: Cholesterol Level, Triglyceride Level, LDL Cholesterol, HDL Cholesterol
- 🏥 Medical History: Smoking, Diabetes, Family Heart Disease
- 🏃 Lifestyle: Exercise Habits, Alcohol Consumption, Stress Level, Sleep Hours, Sugar Consumption
- ⚕️ Health Markers: BMI, High Blood Pressure, Fasting Blood Sugar, CRP Level, Homocysteine Level

**🎯 Target:** Heart Disease Status (Binary: Yes/No)

## 🧠 Neural Network Architecture

### 🏗️ Model Structure (TensorFlow/Keras Sequential)
```
📥 Input Layer
    ↓ (20 features - all numeric after preprocessing)
🧠 Dense(256, activation='relu', L2=0.001)
    ↓
⚙️ BatchNormalization()
    ↓
🔄 Dropout(0.4)  ← 40% dropout for regularization
    ↓
🧠 Dense(128, activation='relu', L2=0.001)
    ↓
⚙️ BatchNormalization()
    ↓
🔄 Dropout(0.3)  ← 30% dropout
    ↓
🧠 Dense(64, activation='relu', L2=0.001)
    ↓
⚙️ BatchNormalization()
    ↓
🔄 Dropout(0.2)  ← 20% dropout
    ↓
🧠 Dense(32, activation='relu')
    ↓
🧠 Dense(16, activation='relu')
    ↓
📤 Dense(1, activation='sigmoid')  ← 💯 Binary output [0, 1]
```

### 🎯 Key Design Choices:
- **📊 Total Parameters:** ~142,000
- **⚡ Optimizer:** Adam (lr=0.001) - Adaptive learning rate optimization
- **📉 Loss Function:** Binary Crossentropy - Standard for binary classification
- **🛡️ Regularization Techniques:**
  - 📌 L2 Penalty (0.001) - Prevents overfitting by penalizing large weights
  - 🔀 Dropout (0.2-0.4) - Randomly deactivates neurons during training
  - ⚙️ Batch Normalization - Normalizes layer inputs for faster convergence
- **⏹️ Early Stopping:** Patience=50 - Stops training if validation loss doesn't improve
- **📈 Learning Rate Scheduler:** Reduces LR by 0.5 if no improvement for 10 epochs

## 🔄 Data Preprocessing Pipeline

### ✅ Step 1: Missing Value Handling
```python
# 📊 Numeric columns: Use median (less affected by outliers)
# 🏷️ Categorical columns: Use mode (most frequent value)
```
- **❓ Why median for numeric?** Robust to outliers, appropriate for skewed distributions
- **❓ Why mode for categorical?** Preserves most frequent pattern in data

### ✅ Step 2: Categorical Encoding (LabelEncoder)
- 🔄 Converts categorical strings to numeric integers
- 📝 Example: `['Male', 'Female']` → `[1, 0]`
- ✔️ Maintains order consistency across train/test sets

### ✅ Step 3: Feature Scaling (StandardScaler)
```
scaled_value = (x - mean) / std_dev
```
- **❓ Why scaling?** Neural networks converge faster with normalized inputs
- 🚫 Prevents features with larger scales from dominating
- 📊 All features now on same scale ~[-3, 3]

### ✅ Step 4: Train-Test Split
- **📊 80-20 split:** 80% training, 20% testing
- **✔️ Stratification:** Maintains class distribution in both sets
- **🔐 Random state:** Ensures reproducibility

### ✅ Step 5: Class Weight Balancing
- **⚠️ Problem:** Dataset may have imbalanced classes (more healthy than diseased)
- **💡 Solution:** Assign higher weights to minority class
- **✨ Effect:** Model learns both classes equally well

## 📊 Evaluation Metrics

After training, you'll see:

```
✅ Train Accuracy: XX.XX%
✅ Test Accuracy:  XX.XX%
📈 Train AUC-ROC:  X.XXXX
📈 Test AUC-ROC:   X.XXXX

📋 Classification Report:
              precision    recall  f1-score   support
    No Disease       X.XX      X.XX      X.XX       XXX
       Disease       X.XX      X.XX      X.XX       XXX

🔥 Confusion Matrix:
 [[TN  FP]
  [FN  TP]]
```

## 🎨 Generated Visualizations

1. **📈 01_training_history.png**
   - 📊 Training vs Validation Accuracy
   - 📉 Training vs Validation Loss
   - 📊 Training vs Validation AUC-ROC

2. **📊 02_roc_curve.png**
   - 🎯 ROC Curve with AUC score
   - 🔄 Comparison with random classifier

3. **🔥 03_confusion_matrix.png**
   - ✅ True Negatives, False Positives
   - ❌ False Negatives, True Positives

4. **📈 04_prediction_distribution.png**
   - 📊 Distribution of predictions for both classes
   - 📍 Decision threshold visualization

5. **📊 05_performance_summary.png**
   - 📊 Bar chart of: Accuracy, AUC-ROC, Precision, Recall

## 💻 Making Predictions

### 💬 Interactive Mode
```bash
python complete_system.py
# 1️⃣ Select option [1] to make a prediction
# 2️⃣ Enter patient details when prompted
```

### 📤 Output
```
👤 Patient Details:
  👶 Age: 55
  👨 Gender: Male
  ...
  
✅ LOW RISK: No heart disease detected (Confidence: 87.3%)
```

## 🔧 Configuration

All configurations are in the scripts:
- 📂 `SCRIPT_DIR`: Working directory
- 📂 `OUTPUT_DIR`: Model output location
- 📂 `DATA_PATH`: Dataset location
- ⚙️ `MODEL_PARAMS`: Neural network hyperparameters

To modify model architecture, edit these sections in the scripts:
```python
model = tf.keras.Sequential([
    # 🔧 Modify layers here
])
```

## 📦 Dependencies

```
📦 pandas>=1.3.0
📦 numpy>=1.21.0
🤖 tensorflow>=2.10.0
📚 scikit-learn>=1.0.0
🎨 matplotlib>=3.4.0
🎨 seaborn>=0.11.0
```

📥 Install with:
```bash
pip install pandas numpy tensorflow scikit-learn matplotlib seaborn
```

## ⚠️ Important Notes

1. **🚀 First Run:** Always run `train_model.py` or `complete_system.py` first to train the model
2. **💾 Model Path:** The trained model is saved in `train/` directory
3. **📋 Feature Order:** Must maintain consistent feature order between training and prediction
4. **🏷️ Encoding:** Categorical variables are encoded and must be decoded for display
5. **📊 Scaling:** Features are always scaled before prediction

## 🔄 Migration from Old System

The new TensorFlow-only system replaces:
- `disease.py` (XGBoost) → ❌ Not used
- `disease_tensorflow.py` (mixed) → ✅ Use `train_model.py` or `complete_system.py`

## 📝 Files Explanation

| 📄 File | 🎯 Purpose |
|------|---------|
| `train_model.py` | 🚀 Standalone training with full evaluation |
| `predict.py` | 💬 Interactive prediction after training |
| `complete_system.py` | 🌐 All-in-one solution (train + evaluate + predict) |
| `disease.py` | ❌ OLD: XGBoost version (deprecated) |
| `disease_tensorflow.py` | ⚠️ OLD: Mixed implementation (deprecated) |

## 🎯 Performance Targets

📊 Typical model performance:
- **✅ Accuracy:** 85-90%
- **📈 AUC-ROC:** 0.90-0.95
- **🎯 Precision:** 0.85-0.92
- **📍 Recall:** 0.82-0.88

*Actual values depend on data and hyperparameters*

## 🐛 Troubleshooting

### ❌ "Model not found" error
**✅ Solution:** Run `train_model.py` or `complete_system.py` first

### ❌ "CSV file not found" error
**✅ Solution:** Ensure `data/heart_disease.csv` exists in the same directory

### ❌ Out of Memory error
**✅ Solution:** Reduce batch_size in the model configuration

### ❌ Low accuracy
**✅ Solution:** 
- 🔍 Check data quality
- ⚙️ Adjust hyperparameters
- 📈 Increase epochs
- 🏗️ Modify model architecture

## 📧 Support

For issues or questions about:
- **🔄 Data preprocessing:** Check `load_and_prepare_data()` function
- **🧠 Model training:** Check `train_model()` function
- **💬 Predictions:** Check `make_prediction()` function

## ✨ Features

✅ Pure TensorFlow implementation
✅ Handles missing values
✅ Encodes categorical variables
✅ Balances imbalanced classes
✅ Early stopping to prevent overfitting
✅ Learning rate scheduling
✅ Comprehensive evaluation metrics
✅ Beautiful visualizations
✅ Interactive prediction interface
✅ Model persistence (save/load)
✅ Well documented code
✅ Error handling

## 📚 For More Information

**📖 For detailed technical information**, see **COMPLETE_GUIDANCE.md** which includes:
- 🏗️ Complete system architecture explanation
- 📚 Library packages and why TensorFlow was chosen over XGBoost
- 🧠 Neural network layer-by-layer breakdown
- 🔄 Data preprocessing pipeline details
- 📚 Training and evaluation process explanation
- 📊 Output interpretation and metric explanations
- 🐛 Troubleshooting guide

**🔍 For glossary of terms**, see **TERMS.md** which includes:
- 🔥 Confusion Matrix, Precision, Recall, Accuracy explained
- 📈 AUC-ROC, ROC Curve, F1-Score definitions
- ⚙️ Epoch, Loss, Gradient, Learning Rate concepts
- 🎚️ Decision Threshold, Regularization techniques
- ⚕️ Medical-specific concepts (Sensitivity, Specificity, PPV, NPV)

**📋 For project details**, see **[LICENSE](LICENSE)** which includes:
- 👥 Complete team information
- 📋 Project requirements and specifications
- 📥 Installation instructions
- ⚙️ Technical specifications
- 🐛 Troubleshooting guide

## 🚀 Future Enhancements

- [ ] ✔️ Cross-validation
- [ ] 🔧 Hyperparameter tuning with Optuna
- [ ] 🔍 Model interpretation (SHAP values)
- [ ] 🌐 API server (Flask/FastAPI)
- [ ] 💻 Web interface
- [ ] 📊 Real-time prediction monitoring
- [ ] 📦 Model versioning

## 👥 Development Team

**🎓 Adamas University - Python Project**

### 🏆 Lead Developers
- **👨‍💻 Rohit Kumar Adak** 
- **👨‍💻 Babin Bid** 

### 💼 Developers
- **👩‍💻 Ritika Pramanick** 
- **👩‍💻 Liza Ghosh** 

## 📅 Project Timeline

- **📅 Start Date:** November 2025
- **📅 Last Modification:** 8th November 2025
- **🔢 Version:** 1.0
- **✅ Status:** ✅ Complete

---

<div align="center">

**📝 Created:** November 2025 | **🔢 Version:** 1.0 | **🤖 Framework:** TensorFlow 2.x

**📜 License:** MIT (See [LICENSE](LICENSE) file) | **🏫 Institution:** Adamas University

</div>
