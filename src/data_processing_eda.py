# src/data_processing_eda.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os # Import os module to handle file paths

# --- Configuration ---
# Define the path to the dataset file
# Using the absolute path provided by the user
DATA_PATH = r"C:\Users\utkar\Desktop\predict+students+dropout+and+academic+success\dataset.csv"
# Define the path for saving EDA plots
# Keeping this relative to the script location
EDA_OUTPUT_PATH = os.path.join('..', 'eda_plots')

# Create the output directory for plots if it doesn't exist
if not os.path.exists(EDA_OUTPUT_PATH):
    os.makedirs(EDA_OUTPUT_PATH)
    print(f"Created directory: {EDA_OUTPUT_PATH}")

# --- Data Loading ---
print(f"Loading data from: {DATA_PATH}")
try:
    # Use a raw string (r"...") or double backslashes (\\) for Windows paths
    # FIX: Added sep=';' because the CSV uses semicolons as separators
    df = pd.read_csv(DATA_PATH, sep=';')
    print("Data loaded successfully.")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Dataset not found at {DATA_PATH}. Please ensure the file path is correct.")
    exit() # Exit the script if the data file is not found
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()

# --- Initial Data Exploration ---
print("\n--- Initial Data Exploration ---")
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nDescriptive Statistics:")
print(df.describe())

print("\nMissing Values per Column:")
print(df.isnull().sum())

# --- Data Cleaning and Transformation ---

# Handle Missing Values (Example: Fill missing numerical with median, categorical with mode)
# Check for columns with missing values
cols_with_missing = df.columns[df.isnull().any()].tolist()
print(f"\nColumns with missing values: {cols_with_missing}")

# Example: If 'SomeNumericalColumn' had missing values, fill with median
# if 'SomeNumericalColumn' in cols_with_missing:
#     median_val = df['SomeNumericalColumn'].median()
#     df['SomeNumericalColumn'].fillna(median_val, inplace=True)
#     print(f"Filled missing values in 'SomeNumericalColumn' with median: {median_val}")

# Example: If 'SomeCategoricalColumn' had missing values, fill with mode
# if 'SomeCategoricalColumn' in cols_with_missing:
#     mode_val = df['SomeCategoricalColumn'].mode()[0]
#     df['SomeCategoricalColumn'].fillna(mode_val, inplace=True)
#     print(f"Filled missing values in 'SomeCategoricalColumn' with mode: {mode_val}")

# Note: Based on the UCI description, this dataset seems to have no missing values.
# If df.isnull().sum() shows 0 for all columns, you can skip the filling steps above.
print("\nMissing values after handling (if any):")
print(df.isnull().sum().sum()) # Check total missing values

# Identify Categorical and Numerical Features
# Based on the dataset description, many columns are categorical or ordinal.
# We need to decide which ones to treat as categorical for encoding.
# Let's assume columns like 'Marital Status', 'Application Mode', 'Course', etc., are categorical.
# The 'Target' column is our target variable.

# Separate features (X) and target (y)
# The target variable is 'Target'
# This drop should now work correctly as 'Target' will be a distinct column
X = df.drop('Target', axis=1)
y = df['Target']

# Identify categorical columns for encoding (excluding the target)
# This list is based on understanding the dataset features. You might need to adjust.
categorical_cols = [col for col in X.columns if X[col].dtype == 'object' or df[col].nunique() < 50] # Heuristic: treat object types or columns with less than 50 unique values as potential categories
# Exclude columns that are clearly numerical identifiers or counts if any
# Example: if 'Student ID' was a column, you'd exclude it.
# Check the actual dataset columns and their meaning from the UCI description.
print(f"\nIdentified potential categorical columns: {categorical_cols}")

# Apply One-Hot Encoding to categorical features
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True) # drop_first=True avoids multicollinearity

print("\nFeatures after One-Hot Encoding:")
print(X.head())
print(f"Shape after encoding: {X.shape}")

# --- Exploratory Data Analysis (EDA) and Visualization ---
print("\n--- Performing EDA and Visualization ---")

# Distribution of the Target Variable
plt.figure(figsize=(8, 6))
sns.countplot(x=y)
plt.title('Distribution of Academic Success (Target)')
plt.xlabel('Target')
plt.ylabel('Count')
plt.savefig(os.path.join(EDA_OUTPUT_PATH, 'target_distribution.png'))
print(f"Saved plot: {os.path.join(EDA_OUTPUT_PATH, 'target_distribution.png')}")
# plt.show() # Uncomment to display plots immediately

# Correlation Matrix (for numerical features)
# Note: After one-hot encoding, we have many numerical features.
# A correlation matrix of all features might be too large.
# Let's look at the correlation of the original numerical features if any, or a subset.
# Identify original numerical columns (before encoding)
original_numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
# Exclude the target variable if it's numerical (in this case, 'Target' is string, but good practice)
if 'Target' in original_numerical_cols:
    original_numerical_cols.remove('Target')

if original_numerical_cols:
    print("\nCorrelation Matrix of Original Numerical Features:")
    correlation_matrix = df[original_numerical_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Original Numerical Features')
    plt.savefig(os.path.join(EDA_OUTPUT_PATH, 'correlation_matrix_numerical.png'))
    print(f"Saved plot: {os.path.join(EDA_OUTPUT_PATH, 'correlation_matrix_numerical.png')}")
    # plt.show()

# Example: Distribution of a few key numerical features (if any meaningful ones exist after encoding)
# Let's pick a few columns that might be interesting, maybe related to grades or age.
# You'll need to check the dataset documentation for meaningful numerical columns.
# Example: Assuming 'Curricular units 1st sem (credited)' and 'Age at enrollment' are important numerical features
# Replace with actual meaningful numerical columns from the dataset if different.
example_numerical_features = ['Curricular units 1st sem (credited)', 'Age at enrollment'] # Replace with actual column names

for col in example_numerical_features:
    if col in df.columns and df[col].dtype in [np.number]:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(EDA_OUTPUT_PATH, f'{col}_distribution.png'))
        print(f"Saved plot: {os.path.join(EDA_OUTPUT_PATH, f'{col}_distribution.png')}")
        # plt.show()
    elif col in X.columns and X[col].dtype in [np.number]:
         # Handle cases where a numerical column might have been created/modified during encoding
         plt.figure(figsize=(8, 6))
         sns.histplot(X[col], kde=True)
         plt.title(f'Distribution of {col} (after processing)')
         plt.xlabel(col)
         plt.ylabel('Frequency')
         plt.savefig(os.path.join(EDA_OUTPUT_PATH, f'{col}_distribution_processed.png'))
         print(f"Saved plot: {os.path.join(EDA_OUTPUT_PATH, f'{col}_distribution_processed.png')}")
         # plt.show()


# Example: Relationship between a categorical feature and the target
# Let's look at 'Marital Status' vs 'Target'
# Replace 'Marital Status' with an actual categorical column you want to visualize.
example_categorical_feature = 'Marital Status' # Replace with actual column name

if example_categorical_feature in df.columns:
    plt.figure(figsize=(10, 7))
    sns.countplot(x=example_categorical_feature, hue='Target', data=df)
    plt.title(f'Academic Success by {example_categorical_feature}')
    plt.xlabel(example_categorical_feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right') # Rotate labels if they overlap
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.savefig(os.path.join(EDA_OUTPUT_PATH, f'{example_categorical_feature}_vs_target.png'))
    print(f"Saved plot: {os.path.join(EDA_OUTPUT_PATH, f'{example_categorical_feature}_vs_target.png')}")
    # plt.show()


print("\nEDA and Data Processing complete. Processed data is stored in the 'X' and 'y' variables.")
print("You can now use these variables for model training.")

# Save the processed data if needed for debugging or later use (optional)
# processed_data = pd.concat([X, y.reset_index(drop=True)], axis=1)
# processed_data.to_csv(os.path.join('..', 'data', 'processed_data.csv'), index=False)
# print(f"Saved processed data to: {os.path.join('..', 'data', 'processed_data.csv')}")

