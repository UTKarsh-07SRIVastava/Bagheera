import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np 

st.set_page_config(page_title="Student Academic Success Predictor", layout="wide")


MODEL_PATH = os.path.join('..', 'models', 'academic_success_model.joblib')
DATA_PATH = r"C:\Users\utkar\Desktop\predict+students+dropout+and+academic+success\dataset.csv"

@st.cache_resource
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

@st.cache_data 
def load_data_and_preprocess(data_path):
    """Loads the original data and identifies categorical columns for encoding."""
    try:

        df = pd.read_csv(data_path, sep=';')

        categorical_cols = [col for col in df.columns if (df[col].dtype == 'object' or df[col].nunique() < 50) and col != 'Target']

        if 'Target' in df.columns:
            X_dummy = df.drop('Target', axis=1)
            X_dummy_encoded = pd.get_dummies(X_dummy, columns=categorical_cols, drop_first=True)
            trained_model_features = X_dummy_encoded.columns.tolist()
        else:
             st.error("Error: 'Target' column not found in the dataset after loading.")
             return None, None, None

        return df, categorical_cols, trained_model_features

    except FileNotFoundError:
        st.error(f"Error: Data file not found at {data_path}. Please ensure the CSV file is in the 'data' folder.")
        st.exception(FileNotFoundError)
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading and processing data for the app: {e}")

        st.exception(e)
        return None, None, None


model = load_model(MODEL_PATH)
original_df, categorical_cols, trained_model_features = load_data_and_preprocess(DATA_PATH)


st.title("Predict Student Academic Success")

st.markdown("""
This application predicts whether a student is likely to **Dropout**, **Enroll**, or **Graduate**
based on their characteristics and academic performance.
""")

if model is not None and original_df is not None and trained_model_features is not None:

    st.header("Enter Student Information")

    input_data = {}

    original_features = original_df.drop('Target', axis=1).columns.tolist()

    for col in original_features:
        if col in categorical_cols:
            unique_values = original_df[col].unique().tolist()
            unique_values = sorted([str(val) for val in unique_values if pd.notna(val)])
            input_data[col] = st.selectbox(f"Select {col}", unique_values)
        elif np.issubdtype(original_df[col].dtype, np.number):
            min_val = float(original_df[col].min()) if pd.notna(original_df[col].min()) else 0.0
            max_val = float(original_df[col].max()) if pd.notna(original_df[col].max()) else 100.0 # Adjust max as needed
            mean_val = float(original_df[col].mean()) if pd.notna(original_df[col].mean()) else (min_val + max_val) / 2
            input_data[col] = st.number_input(f"Enter {col}", min_value=min_val, max_value=max_val, value=mean_val, step=0.01)
        else:

            input_data[col] = st.text_input(f"Enter {col}")

    if st.button("Predict Academic Success"):
        input_df = pd.DataFrame([input_data])

        input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

        for col in trained_model_features:
            if col not in input_df_encoded.columns:
                input_df_encoded[col] = 0 
        input_df_encoded = input_df_encoded[trained_model_features]

        try:
            prediction = model.predict(input_df_encoded)
            prediction_proba = model.predict_proba(input_df_encoded)

            st.subheader("Prediction Result:")
            st.write(f"The predicted academic success status is: **{prediction[0]}**")
            st.subheader("Prediction Probabilities:")
            proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
            st.dataframe(proba_df)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    st.warning("Application is unable to load data or model. Please check the file paths and ensure previous steps were successful.")

st.markdown("---")
st.markdown("Built with Streamlit and Scikit-learn")
