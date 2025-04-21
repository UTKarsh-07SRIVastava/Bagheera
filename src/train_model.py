# src/train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # A good choice for classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # To save the model
import os

# --- Configuration ---
# Define the path to the dataset file
# Using the absolute path provided by the user
DATA_PATH = r"C:\Users\utkar\Desktop\predict+students+dropout+and+academic+success\dataset.csv"
MODEL_PATH = os.path.join('..', 'models', 'academic_success_model.joblib')

# Create the models directory if it doesn't exist
if not os.path.exists(os.path.join('..', 'models')):
    os.makedirs(os.path.join('..', 'models'))
    print(f"Created directory: {os.path.join('..', 'models')}")

# --- Data Loading and Preprocessing (Repeat from data_processing_eda.py for a self-contained script) ---
print(f"Loading data for model training from: {DATA_PATH}")
try:
    # Use a raw string (r"...") or double backslashes (\\) for Windows paths
    # FIX: Added sep=';' because the CSV uses semicolons as separators
    df = pd.read_csv(DATA_PATH, sep=';')
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: Dataset not found at {DATA_PATH}. Please ensure the file path is correct.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()

# Separate features (X) and target (y)
X = df.drop('Target', axis=1)
y = df['Target']

# Identify and One-Hot Encode categorical columns (must be consistent with EDA script)
categorical_cols = [col for col in X.columns if X[col].dtype == 'object' or df[col].nunique() < 50]
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print("\nProcessed data shape for model training:", X.shape)

# --- Model Training ---
print("\n--- Model Training ---")

# Split data into training and testing sets
# Using a stratify split is good for classification to maintain target distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Initialize the model
# RandomForestClassifier is a robust choice, often performs well out-of-the-box.
# You could experiment with other models like Logistic Regression, SVM, Gradient Boosting, etc.
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # class_weight='balanced' helps with imbalanced datasets

# Train the model
print("Training the model...")
model.fit(X_train, y_train)
print("Model training complete.")

# --- Model Evaluation ---
print("\n--- Model Evaluation ---")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Model Export ---
print("\n--- Model Export ---")

# Save the trained model to a file
try:
    joblib.dump(model, MODEL_PATH)
    print(f"Model successfully saved to {MODEL_PATH}")
except Exception as e:
    print(f"Error saving the model: {e}")

print("\nModel training and export complete.")

