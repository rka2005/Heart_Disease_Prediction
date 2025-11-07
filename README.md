# Heart Disease Prediction System

This project uses a scikit-learn AdaBoost classifier to predict the risk of heart disease based on key health parameters. The system analyzes 7 key features to provide fast and accurate risk assessments.

## Features Used for Prediction

The model considers all available health indicators from the dataset, including:

1. **Age (years)**: Age in years
2. **BMI (Body Mass Index)**: Automatically calculated from weight and height inputs using the standard formula: BMI = weight(kg) / [height(meters)]²
3. **Cholesterol Level (mg/dL)**: Total cholesterol
4. **Blood Pressure (mmHg)**: Blood pressure measurement
5. **CRP Level (C-Reactive Protein, mg/L)**: Inflammation marker
6. **Smoking**: Smoking status (Yes/No)
7. **Diabetes**: Diabetes status (Yes/No)
8. **And additional features**: Gender, Stress Level, Family Heart Disease, Alcohol Consumption, Sugar Consumption, etc.

## Project Structure

- `disease.py`: Trains the AdaBoost model, preprocesses data, and saves the trained model and scaler.
- `predict_gui.py`: User-friendly GUI application for inputting health data and getting instant predictions.
- `data/`: Contains the original dataset and preprocessed version.
  - `heart_disease.csv`: Raw dataset with all features.
  - `preprocessed_heart_disease.csv`: Cleaned dataset with all features.
- `models/`: Stores trained models and related files.
  - `ada_heart_model.pkl`: Trained scikit-learn AdaBoost model.
  - `heart_scaler_7param.pkl`: Scaler for feature normalization.
  - `ada_feature_importances.png`: Feature importance visualization.

## Setup Instructions

1. **Prerequisites**: Python 3.7+ installed.
2. **Install Dependencies**:
   ```bash
   pip install pandas scikit-learn matplotlib joblib
   ```
3. **Data Preparation**: Ensure `data/heart_disease.csv` is present (already included).

## Usage Guide

### Training the Model
1. Open command prompt in the project directory.
2. Run: `python disease.py`
3. The script will:
   - Load and preprocess the data
   - Train the AdaBoost model
   - Save the model and scaler in `models/`
   - Display evaluation metrics (accuracy, precision, recall)
   - Generate feature importance plot

### Making Predictions
1. Run: `python predict_gui.py`
2. A GUI window will open with input fields for key features.
3. Fill in your health data:
   - Enter numerical values for Age, Cholesterol, Blood Pressure, CRP Level
   - Select Yes/No for Smoking and Diabetes
   - **BMI Calculation**: Enter your Weight (kg) and Height (feet and inches)
     - The system automatically calculates BMI using: `BMI = weight(kg) / [height(meters)]²`
     - Height conversion: `total_meters = (feet × 0.3048) + (inches × 0.0254)`
     - **Example**: 5 feet 10 inches = (5 × 0.3048) + (10 × 0.0254) = 1.778 meters
     - **Example**: 75 kg person, 1.778m tall → BMI = 75 / (1.778)² = 23.7
4. Click "Predict" to get your risk assessment.
5. Results show risk level (High/Low) with confidence percentage.

## Model Details

- **Algorithm**: Scikit-learn AdaBoost Classifier
- **Architecture**: 200 decision stumps with learning rate 0.1
- **Input**: 7 key health features (matches GUI inputs)
- **Output**: Probability of heart disease (0-1)
- **Threshold**: 45% probability triggers "High Risk" warning
- **Training**: Uses stratified split and adaptive boosting
- **Accuracy**: 80.0% on test set
- **Speed**: Fast training and inference

## Model Performance Comparison

| Model | Accuracy | Features | Notes |
|-------|----------|----------|-------|
| **AdaBoost** | **80.0%** | **7** | 🏆 **FINAL MODEL** - GUI compatible |
| XGBoost (Tuned) | 76.2% | 20 | Strong alternative |
| Random Forest | 75.1% | 20 | Good ensemble method |
| LightGBM | 71.7% | 20 | Fast gradient boosting |
| Voting (Soft) | 74.8% | 20 | Ensemble approach |
| MLP Neural Net | 69.3% | 20 | Deep learning |
| Logistic Regression | 51.4% | 20 | Poor linear fit |

## Understanding Results

- **High Risk**: Probability > 45% - Indicates potential heart disease. Consult a healthcare professional.
- **Low Risk**: Probability ≤ 45% - Lower likelihood, but maintain healthy lifestyle.
- **Confidence**: Shows the model's certainty in the prediction.

## Important Notes

- This is a screening tool, not a medical diagnosis.
- Always consult healthcare providers for proper medical advice.
- The model is trained on general population data; individual results may vary.
- Regular health check-ups are recommended regardless of prediction results.
- **Data Quality Limitation**: Current dataset has weak correlations, limiting maximum accuracy to ~80%.

## Troubleshooting

- **Import Errors**: Ensure all packages are installed correctly.
- **Model Loading Issues**: Make sure `disease.py` has been run to create the model files.
- **GUI Not Opening**: Check if tkinter is available (usually included with Python).

For questions or improvements, feel free to modify the code!