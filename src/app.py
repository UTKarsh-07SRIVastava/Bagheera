# src/app.py

import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np # Import numpy for numerical operations

# --- Streamlit App Layout ---
# FIX: Moved set_page_config to the very top, as it must be the first Streamlit command
st.set_page_config(page_title="Student Academic Success Predictor", layout="wide")


# --- Configuration ---
# Define the path to the trained model file
MODEL_PATH = os.path.join('..', 'models', 'academic_success_model.joblib')
# Define the path to the original data file (needed to get column names and categories for encoding)
# Using the absolute path provided by the user
DATA_PATH = r"C:\Users\utkar\Desktop\predict+students+dropout+and+academic+success\dataset.csv"


# --- Load Model and Data ---
@st.cache_resource # Cache the model loading for better performance
def load_model(model_path):
    """Loads the trained model."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}. Please ensure train_model.py was run successfully.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

@st.cache_data # Cache the data loading and preprocessing
def load_data_and_preprocess(data_path):
    """Loads the original data and identifies categorical columns for encoding."""
    try:
        # FIX: Added sep=';' because the CSV uses semicolons as separators
        df = pd.read_csv(data_path, sep=';')

        # Identify categorical columns using the same logic as in training
        # Ensure 'Target' is not included in categorical_cols
        categorical_cols = [col for col in df.columns if (df[col].dtype == 'object' or df[col].nunique() < 50) and col != 'Target']

        # Get the list of columns after one-hot encoding from a dummy run
        # This is crucial to ensure the input features for prediction match the training features
        # Check if 'Target' column exists before dropping
        if 'Target' in df.columns:
            X_dummy = df.drop('Target', axis=1)
            X_dummy_encoded = pd.get_dummies(X_dummy, columns=categorical_cols, drop_first=True)
            trained_model_features = X_dummy_encoded.columns.tolist()
        else:
             st.error("Error: 'Target' column not found in the dataset after loading.")
             return None, None, None # Return None if Target column is missing


        # Return the original dataframe (for input widgets) and the list of feature names the model expects
        return df, categorical_cols, trained_model_features

    except FileNotFoundError:
        st.error(f"Error: Data file not found at {data_path}. Please ensure the CSV file is in the 'data' folder.")
        # Add more specific error logging for debugging
        st.exception(FileNotFoundError)
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading and processing data for the app: {e}")
        # Add more specific error logging for debugging
        st.exception(e)
        return None, None, None


model = load_model(MODEL_PATH)
original_df, categorical_cols, trained_model_features = load_data_and_preprocess(DATA_PATH)


st.title("Predict Student Academic Success")

st.markdown("""
This application predicts whether a student is likely to **Dropout**, **Enroll**, or **Graduate**
based on their characteristics and academic performance.
""")

# Only proceed with input form and prediction if data and model loaded successfully
if model is not None and original_df is not None and trained_model_features is not None:

    # --- Input Form ---
    st.header("Enter Student Information")

    # Create input widgets for each feature
    # We need to create widgets that correspond to the features in the *original* dataset
    # before one-hot encoding, and then manually encode the user's input.

    input_data = {}

    # Get the original column names (excluding the target)
    # This line is now inside the check for original_df not being None
    original_features = original_df.drop('Target', axis=1).columns.tolist()

    # Create input widgets based on the data types and unique values in the original data
    for col in original_features:
        if col in categorical_cols:
            # Use selectbox for categorical features
            unique_values = original_df[col].unique().tolist()
            # Sort unique values for better presentation, handle potential NaNs if any
            unique_values = sorted([str(val) for val in unique_values if pd.notna(val)])
            input_data[col] = st.selectbox(f"Select {col}", unique_values)
        # FIX: Use np.issubdtype to check for numerical types
        elif np.issubdtype(original_df[col].dtype, np.number):
            # Use number_input for numerical features
            min_val = float(original_df[col].min()) if pd.notna(original_df[col].min()) else 0.0
            max_val = float(original_df[col].max()) if pd.notna(original_df[col].max()) else 100.0 # Adjust max as needed
            mean_val = float(original_df[col].mean()) if pd.notna(original_df[col].mean()) else (min_val + max_val) / 2
            input_data[col] = st.number_input(f"Enter {col}", min_value=min_val, max_value=max_val, value=mean_val, step=0.01)
        else:
            # Default to text input for other types
            input_data[col] = st.text_input(f"Enter {col}")

    # --- Prediction ---
    if st.button("Predict Academic Success"):
        # Convert input data to a pandas DataFrame
        input_df = pd.DataFrame([input_data])

        # Apply the same one-hot encoding as used during training
        # Crucially, we need to reindex the columns to match the training data columns
        input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

        # Add missing columns that were in the training data but not in the input (due to one-hot encoding)
        # and ensure the order of columns is the same as the training data
        for col in trained_model_features:
            if col not in input_df_encoded.columns:
                input_df_encoded[col] = 0 # Add missing columns with a value of 0

        # Ensure the columns are in the same order as the training data
        input_df_encoded = input_df_encoded[trained_model_features]


        # Make prediction
        try:
            prediction = model.predict(input_df_encoded)
            prediction_proba = model.predict_proba(input_df_encoded)

            st.subheader("Prediction Result:")
            st.write(f"The predicted academic success status is: **{prediction[0]}**")

            # Display prediction probabilities
            st.subheader("Prediction Probabilities:")
            proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
            st.dataframe(proba_df)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    # Display a message if data or model failed to load
    st.warning("Application is unable to load data or model. Please check the file paths and ensure previous steps were successful.")


# --- Footer ---
st.markdown("---")
st.markdown("Built with Streamlit and Scikit-learn")

