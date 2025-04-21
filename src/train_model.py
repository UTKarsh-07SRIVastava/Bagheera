import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
DATA_PATH = r"C:\Users\utkar\Desktop\predict+students+dropout+and+academic+success\dataset.csv"
MODEL_PATH = os.path.join('..', 'models', 'academic_success_model.joblib')
if not os.path.exists(os.path.join('..', 'models')):
    os.makedirs(os.path.join('..', 'models'))
    print(f"Created directory: {os.path.join('..', 'models')}")
print(f"Loading data for model training from: {DATA_PATH}")
try:
    df = pd.read_csv(DATA_PATH, sep=';')
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: Dataset not found at {DATA_PATH}. Please ensure the file path is correct.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()
X = df.drop('Target', axis=1)
y = df['Target']
categorical_cols = [col for col in X.columns if X[col].dtype == 'object' or df[col].nunique() < 50]
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
print("\nProcessed data shape for model training:", X.shape)
print("\n--- Model Training ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') 
print("Training the model...")
model.fit(X_train, y_train)
print("Model training complete.")

print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n--- Model Export ---")

try:
    joblib.dump(model, MODEL_PATH)
    print(f"Model successfully saved to {MODEL_PATH}")
except Exception as e:
    print(f"Error saving the model: {e}")

print("\nModel training and export complete.")
