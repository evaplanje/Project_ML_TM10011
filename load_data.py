
#%% Imports

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#%% Loading data from a CSV-file

def load_data(file_name):
    """
    Parameters
    ----------
    file_name : str
        Name of the CSV-file (for example: 'data.csv').
    
    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with index

    """
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_directory, file_name)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{file_name} not found in {this_directory}")
    
    return pd.read_csv(data_path, index_col=0)

#%% Explore data to find missing values, duplicate rows and class-counts

def explore_data(df):
    
    print("\nMissing Values (Top 10):")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing.head(10) if not missing.empty else "No missing values found.")
    
    print("\nDuplicate Rows:", df.duplicated().sum())

    labels = df.iloc[:, 0]
    counts = labels.value_counts()
    print(f"\nClass counts:\n{counts}\n")

    for i, col in enumerate(df.columns, 1):
        print(f"{i:3d}. {col}")

#%% Split the dataset into a train- and testset

def split_pd(df, show_details = True):
    """
    Splits a DataFrame into training- and testset (80/20) based on the 'label' column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the features and the 'label' column.

    Returns
    -------
    X_train : pd.DataFrame
        Feature DataFrame for training
    X_test : pd.DataFrame
        Feature DataFrame for testing
    y_train : pd.Series
        Training labels
    y_test : pd.Series
        Test labels
    """
    RANDOM_STATE = 42 #Random state to ensure the same split each time

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,          
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y
    )
    y_train = y_train.map({'non-GIST': 0, 'GIST': 1})
    y_test = y_test.map({'non-GIST': 0, 'GIST': 1})

    if show_details == True:
        print("Train size:", X_train.shape[0])
        print("Test size:", X_test.shape[0])

        print("\nTrain class distribution:")
        print(y_train.value_counts(normalize=True))

        print("\nTest class distribution:")
        print(y_test.value_counts(normalize=True))
    return X_train, X_test, y_train, y_test
