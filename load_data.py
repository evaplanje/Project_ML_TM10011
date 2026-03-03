#%%

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


#%%

def load_data(file_name):
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_directory, file_name)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{file_name} not found in {this_directory}")
    
    return pd.read_csv(data_path, index_col=0)

def explore_data(df):
    print("\nGIST data")
    
    print(f"\nShape: {df.shape}")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nColumn Types:")
    print(df.dtypes.value_counts())
    
    print("\nMissing Values (Top 10):")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing.head(10) if not missing.empty else "No missing values found.")
    
    print("\nDuplicate Rows:", df.duplicated().sum())

    labels = df.iloc[:, 0]
    counts = labels.value_counts()
    print(f"\nClass counts:\n{counts}\n")



    for i, col in enumerate(GIST_data.columns, 1):
        print(f"{i:3d}. {col}")
        # if i > 10:
            # break

def plot_feature_pairs(df):
    le = LabelEncoder()
    y = le.fit_transform(df.iloc[:, 0])     # df.iloc[:, 0] bevat label GIST en non-GIST (en wordt 0 = non-GIST, 1 = GIST)
    X = df.iloc[:, 1:]                      

    fig = plt.figure(figsize=(18, 5))
    
    ax1 = fig.add_subplot(131)
    ax1.scatter(X.iloc[:, 0], X.iloc[:, 1],
                c=y, cmap='viridis', edgecolor='k', s=40)
    ax1.set_title('plot 1')
    ax1.set_xlabel(X.columns[0])
    ax1.set_ylabel(X.columns[1])

    ax2 = fig.add_subplot(132)
    ax2.scatter(X.iloc[:, 2], X.iloc[:, 3],
                c=y, cmap='viridis', edgecolor='k', s=40)
    ax2.set_title('plot 2')
    ax2.set_xlabel(X.columns[2])
    ax2.set_ylabel(X.columns[3])
    
    ax3 = fig.add_subplot(133)
    ax3.scatter(X.iloc[:, 4], X.iloc[:, 5],
                c=y, cmap='viridis', edgecolor='k', s=40)
    ax3.set_title('plot 3')
    ax3.set_xlabel(X.columns[4])
    ax3.set_ylabel(X.columns[5])
    plt.tight_layout()
    plt.show()

def plot_heatmap(df, size=20):
    numeric_features = df.select_dtypes(include=['float64', 'int64']).iloc[:, :size]
    corr_matrix = numeric_features.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f'Correlation Heatmap of the First {size} Features')
    plt.show() 

def split_pd(df):

    RANDOM_STATE = 42

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

    print("Train size:", X_train.shape[0])
    print("Test size:", X_test.shape[0])

    print("\nTrain class distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTest class distribution:")
    print(y_test.value_counts(normalize=True))
    return X_train, X_test, y_train, y_test


#%%

GIST_data = load_data('GIST_radiomicFeatures.csv')

explore_data(GIST_data)

plot_feature_pairs(GIST_data)

plot_heatmap(GIST_data)

GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data)
