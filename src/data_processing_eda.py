import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
DATA_PATH = r"C:\Users\utkar\Desktop\predict+students+dropout+and+academic+success\dataset.csv"
EDA_OUTPUT_PATH = os.path.join('..', 'eda_plots')

if not os.path.exists(EDA_OUTPUT_PATH):
    os.makedirs(EDA_OUTPUT_PATH)
    print(f"Created directory: {EDA_OUTPUT_PATH}")
print(f"Loading data from: {DATA_PATH}")
try:
    df = pd.read_csv(DATA_PATH, sep=';')
    print("Data loaded successfully.")
    print(f"Dataset shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Dataset not found at {DATA_PATH}. Please ensure the file path is correct.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()

print("\n--- Initial Data Exploration ---")
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nDescriptive Statistics:")
print(df.describe())

print("\nMissing Values per Column:")
print(df.isnull().sum())
cols_with_missing = df.columns[df.isnull().any()].tolist()
print(f"\nColumns with missing values: {cols_with_missing}")
print("\nMissing values after handling (if any):")
print(df.isnull().sum().sum()) 
X = df.drop('Target', axis=1)
y = df['Target']
categorical_cols = [col for col in X.columns if X[col].dtype == 'object' or df[col].nunique() < 50] 
print(f"\nIdentified potential categorical columns: {categorical_cols}")
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True) 
print("\nFeatures after One-Hot Encoding:")
print(X.head())
print(f"Shape after encoding: {X.shape}")
print("\n--- Performing EDA and Visualization ---")
plt.figure(figsize=(8, 6))
sns.countplot(x=y)
plt.title('Distribution of Academic Success (Target)')
plt.xlabel('Target')
plt.ylabel('Count')
plt.savefig(os.path.join(EDA_OUTPUT_PATH, 'target_distribution.png'))
print(f"Saved plot: {os.path.join(EDA_OUTPUT_PATH, 'target_distribution.png')}")
original_numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
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
example_numerical_features = ['Curricular units 1st sem (credited)', 'Age at enrollment']

for col in example_numerical_features:
    if col in df.columns and df[col].dtype in [np.number]:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(EDA_OUTPUT_PATH, f'{col}_distribution.png'))
        print(f"Saved plot: {os.path.join(EDA_OUTPUT_PATH, f'{col}_distribution.png')}")
    elif col in X.columns and X[col].dtype in [np.number]:
         plt.figure(figsize=(8, 6))
         sns.histplot(X[col], kde=True)
         plt.title(f'Distribution of {col} (after processing)')
         plt.xlabel(col)
         plt.ylabel('Frequency')
         plt.savefig(os.path.join(EDA_OUTPUT_PATH, f'{col}_distribution_processed.png'))
         print(f"Saved plot: {os.path.join(EDA_OUTPUT_PATH, f'{col}_distribution_processed.png')}")
example_categorical_feature = 'Marital Status'
if example_categorical_feature in df.columns:
    plt.figure(figsize=(10, 7))
    sns.countplot(x=example_categorical_feature, hue='Target', data=df)
    plt.title(f'Academic Success by {example_categorical_feature}')
    plt.xlabel(example_categorical_feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_PATH, f'{example_categorical_feature}_vs_target.png'))
    print(f"Saved plot: {os.path.join(EDA_OUTPUT_PATH, f'{example_categorical_feature}_vs_target.png')}")
    # plt.show()


print("\nEDA and Data Processing complete. Processed data is stored in the 'X' and 'y' variables.")
print("You can now use these variables for model training.")
