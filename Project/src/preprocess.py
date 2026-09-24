# src/preprocess.py

import pandas as pd

def load_data(path):
    """Load the raw insurance CSV file."""
    df = pd.read_csv(path)
    return df

def clean_data(df):
    """Remove duplicates, handle any missing values (if present)."""
    df = df.drop_duplicates()
    df = df.dropna()  # only relevant if you actually found missing values in EDA
    return df

def preprocess_data(df):
    """
    Encodes categorical variables and creates engineered features.
    This function is used both during training AND at prediction time,
    so both places transform data identically.
    """
    df = df.copy()

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

    # Optional feature engineering (only include if you actually kept these in your notebook)
    # df['smoker_bmi_interaction'] = df['smoker_yes'] * df['bmi']

    return df