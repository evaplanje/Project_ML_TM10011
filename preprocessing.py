#%%

import pandas as pd
import os
import numpy as np
from load_data import load_data, split_pd, explore_data
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import RobustScaler

#%% 

def apply_normalization(df):
    """
    It centers the data and scales it based on the 25th and 75th percentiles
    
    Parameters
    ----------
    pd.DataFrame
    
    Returns
    -------
    pd.DataFrame
        normalized dataframe
    """
    scaler = RobustScaler()
    scaled = scaler.fit_transform(df)

    # Create a new DataFrame with the normalised feature values
    df_normalized = pd.DataFrame(scaled, columns=df.columns, index=df.index)
    return df_normalized, scaler

def remove_zero_variance_features(df, show_details=True):
    """
    Remove features with zero variance (the features are identical for all samples)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with zero-variance columns removed
    Index
        Remaining feature names
    """
    # Find all features with zero variance 
    zero_var_cols = df.columns[~(df != df.iloc[0]).any()]
    
    # Create a new DataFrame with the kept features
    df_reduced = df.drop(columns=zero_var_cols)
    kept_features = df_reduced.columns

    if show_details:
        print(f"Number of features before: {df.shape[1]}")
        print(f"Dropped zero variance features: {len(zero_var_cols)}")
        print(f"Remaining features: {len(kept_features)}")

    return df_reduced, kept_features

def remove_highly_correlated_features(df, correlation_threshold=0.95, show_details=False):
    """
    Remove features that are highly correlated with another feature

    Parameters
    ----------
    df : pd.DataFrame
    correlation_threshold : float

    Returns
    -------
    pd.DataFrame
        DataFrame with correlated features removed
    Index
        Remaining feature names
    """
    # Create a correlation matrix 
    corr_matrix = df.corr().abs()

    # Extract the upper part of the correlation matrix to avoid duplicate pairs
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Remove features with a correlation above the threshold
    corr_drop_cols = [
        column for column in upper.columns
        if any(upper[column] > correlation_threshold)
    ]

    # Create a new DataFrame with the kept features
    df_reduced = df.drop(columns=corr_drop_cols)
    kept_features = df_reduced.columns

    if show_details:
        print(f"Number of features before: {df.shape[1]}")
        print(f"Dropped highly correlated features: {len(corr_drop_cols)}")
        print(f"Remaining features: {len(kept_features)}")

    return df_reduced, kept_features

#%%