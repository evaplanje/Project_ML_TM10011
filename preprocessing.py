#%%

import pandas as pd
import os
import numpy as np
from load_data import load_data, split_pd, explore_data, plot_feature_pairs, plot_heatmap
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import RobustScaler

#%% 

def apply_normalization(df):
    """
    It centers your data and scales it based on the 25th and 75th percentiles.
    
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
    df_normalized = pd.DataFrame(scaled, columns=df.columns, index=df.index)
    return df_normalized, scaler

def remove_zero_variance_features(df, show_details=True):
    """
    Remove features with zero variance (all values identical).

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

    zero_var_cols = df.columns[~(df != df.iloc[0]).any()]
    df_reduced = df.drop(columns=zero_var_cols)
    kept_features = df_reduced.columns

    if show_details:
        print(f"Number of features before: {df.shape[1]}")
        print(f"Dropped zero variance features: {len(zero_var_cols)}")
        print(f"Remaining features: {len(kept_features)}")

    return df_reduced, kept_features

def remove_highly_correlated_features(df, correlation_threshold=0.90, show_details=False):
    """
    Remove features that are highly correlated with another feature.

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

    corr_matrix = df.corr().abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    corr_drop_cols = [
        column for column in upper.columns
        if any(upper[column] > correlation_threshold)
    ]

    df_reduced = df.drop(columns=corr_drop_cols)
    kept_features = df_reduced.columns

    if show_details:
        print(f"Number of features before: {df.shape[1]}")
        print(f"Dropped highly correlated features: {len(corr_drop_cols)}")
        print(f"Remaining features: {len(kept_features)}")

    return df_reduced, kept_features

#%%

GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)

#%%

normalized_GIST_train, scaler = apply_normalization(GIST_train)
preproc_GIST_train, kept_features = remove_zero_variance_features(normalized_GIST_train, show_details=False)
preproc_GIST_train, kept_features = remove_highly_correlated_features(preproc_GIST_train, correlation_threshold=0.90, show_details=False)
