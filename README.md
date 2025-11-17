# ❤️ Heart Disease Prediction Using TensorFlow

A machine learning project that predicts the risk of heart disease in patients using a deep neural network built with TensorFlow/Keras. This project was developed as a university assignment in the Computer Science and Engineering (CSE) department.

## 📋 Project Overview

This project implements a binary classification model to predict whether a patient is at risk of heart disease based on various health metrics and lifestyle factors. The model uses advanced techniques including:

- **Data Preprocessing**: Handling missing values using median/mode imputation.
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique).
- **Neural Network Architecture**: Deep learning with batch normalization and dropout regularization.
- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Live Prediction**: Interactive command-line interface for real-time predictions.

## 👥 Team Members

- **Babin Bid** - 👨‍💻 Lead Developer
- **Rohit Kumar Adak** - 👨‍💻 Lead Developer
- **Ritika Pramanick** - 👩‍💻 Developer
- **Liza Ghosh** - 👩‍💻 Developer

**Mentor**: 👩‍🏫 Dr. Debdutta Pal

**Project Timeline**: 📅 November 3, 2025 - November 11, 2025

## 🏗️ Project Structure

```
Heart_Disease_Prediction/
├── disease_tensorflow.py          # Main training and prediction script
├── data/
│   └── heart_disease.csv          # Dataset with 10,000+ patient records
├── train/
│   ├── tf_heart_model_full_features.keras       # Trained model weights
│   ├── tf_improved_accuracy.png                  # Training accuracy plot
│   ├── tf_improved_loss.png                      # Training loss plot
│   └── prediction_confidence.png                 # Prediction confidence visualization
├── README.md                       # This file
├── LICENSE                         # MIT License
└── .gitignore                      # Git ignore rules
```

## 📊 Dataset

The dataset contains **10,000+ patient records** with 21 features including:

### Health Metrics
- 🧓 Age
- 🚹🚺 Gender
- 🩸 Blood Pressure
- 🧪 Cholesterol Level
- ⚖️ BMI (Body Mass Index)
- 🍬 Fasting Blood Sugar
- 🧫 Triglyceride Level
- 🧬 CRP Level
- 🧪 Homocysteine Level

### Lifestyle Factors
- 🏃‍♂️ Exercise Habits
- 🚬 Smoking Status
- 🍺 Alcohol Consumption
- 😰 Stress Level
- 😴 Sleep Hours
- 🍭 Sugar Consumption

### Medical History
- 👨‍👩‍👧‍👦 Family Heart Disease
- 💉 Diabetes Status
- 🩸 High Blood Pressure
- 🧪 Low HDL Cholesterol
- 🧪 High LDL Cholesterol

**Target Variable**: ❤️ Heart Disease Status (Yes/No)

## 🔧 Technical Stack

- **🐍 Python 3.12+**
- **🤖 TensorFlow/Keras** - Deep learning framework
- **📊 Pandas** - Data manipulation and analysis
- **🔢 NumPy** - Numerical computing
- **🧠 Scikit-learn** - Machine learning utilities and metrics
- **⚖️ Imbalanced-learn** - SMOTE for handling class imbalance
- **📈 Matplotlib** - Data visualization

## 🚀 Getting Started

### Prerequisites

Install the required packages:

```bash
pip install pandas numpy scikit-learn tensorflow imbalanced-learn matplotlib
```

### Running the Project

1. **Navigate to the project directory:**
   ```bash
   cd Heart_Disease_Prediction
   ```

2. **Run the training script:**
   ```bash
   python disease_tensorflow.py
   ```

3. **Follow the interactive prompts:**
   - Enter patient health metrics when prompted
   - View real-time prediction results with confidence scores
   - Generated visualizations are saved in the `train/` directory

## 🧠 Model Architecture

The neural network consists of:

```
Input Layer (Features) → Dense(128, ReLU) → BatchNorm → Dropout(0.3)
                      → Dense(64, ReLU) → BatchNorm → Dropout(0.2)
                      → Dense(32, ReLU)
                      → Dense(1, Sigmoid) → Output (0-1 probability)
```

**Key Features:**
- **Adam Optimizer** with learning rate of 0.0005
- **Binary Crossentropy** loss function
- **Metrics**: Accuracy, Precision, Recall, AUC
- **Early Stopping**: Monitors validation loss with patience of 25 epochs
- **Training**: Up to 200 epochs on SMOTE-balanced data

## 📈 Results

The trained model provides:
- **Accuracy Score**: Evaluated on test set
- **Precision & Recall**: For both disease and non-disease cases
- **Confusion Matrix**: For detailed performance analysis
- **Training Plots**: Accuracy and loss curves for both training and validation sets
- **Confidence Visualization**: Per-prediction probability distribution

## 🔍 Key Features

### Data Preprocessing
- 📊 Median imputation for numerical features
- 📈 Mode imputation for categorical features
- 🏷️ Label encoding for categorical variables
- 📏 StandardScaler normalization

### Class Imbalance Handling
- ⚖️ SMOTE applied to training data to balance classes
- 🚫 Prevents model bias towards the majority class

### Model Training
- ✂️ 80-20 train-test split with stratification
- 📊 15% validation split during training
- 🛑 Early stopping to prevent overfitting
- 📦 Batch size of 32 samples per iteration

### Live Prediction
- 💬 Interactive input for all 20 features
- 🔄 Encoded categorical inputs (e.g., "Low", "Medium", "High")
- 📊 Real-time confidence percentages
- 📊 Visual confidence distribution chart

## 📝 Input Guide for Predictions

When running live predictions, you'll be prompted for:

1. **🔢 Numerical Values**: Age, Blood Pressure, Cholesterol, etc. (enter as numbers)
2. **📝 Categorical Options**: Gender (Male/Female), Smoking (Yes/No), etc.
3. **😰 Stress Level**: Enter 1 (Low), 2 (Medium), or 3 (High)
4. **🏃‍♂️ Exercise Habits**: High, Low, or Medium

## 📊 Output Files

The script generates the following files in the `train/` directory:

- `tf_heart_model_full_features.keras` - Serialized trained model
- `tf_improved_accuracy.png` - Training vs validation accuracy plot
- `tf_improved_loss.png` - Training vs validation loss plot
- `prediction_confidence.png` - Confidence distribution for current prediction

## ⚠️ Important Notes

- 🔬 The model is trained on the provided dataset and should be validated on external datasets for production use
- 📊 Missing values are handled using median/mode imputation
- 📏 All numerical features are standardized before feeding to the model
- 🏷️ Categorical features are label-encoded during preprocessing
- ⚖️ SMOTE is applied only to the training set to prevent data leakage

## 📚 Dependencies

```
pandas
numpy
scikit-learn
tensorflow
imbalanced-learn
matplotlib
```

## 🎓 Educational Value

This project demonstrates:
- 🔄 End-to-end machine learning pipeline
- 🤖 Deep learning with TensorFlow/Keras
- ⚖️ Handling class imbalance in medical datasets
- 🔍 Cross-validation and hyperparameter tuning
- 📊 Model evaluation and visualization
- 💬 Interactive prediction system

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

This is a university project completed as of November 11, 2025. For modifications or improvements, please contact the development team.

## 📞 Contact

For questions or clarifications regarding this project, please reach out to the development team or 👩‍🏫 Dr. Debdutta Pal (mentor).

---

**Project Completion Date**: November 11, 2025  
**Department**: Computer Science and Engineering (CSE)  
**University**: Adamas University
