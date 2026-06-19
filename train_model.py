import joblib
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer

# Load the core dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Train a baseline XGBoost model
model = XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X, y)

# Ensure the directory D:\models exists on your system, or update the file path
# Save the trained model file
joblib.dump(model, r'D:\models\xgboost_disease_model.pkl')
print("Model saved successfully! Ready for Streamlit application.")