#%%

import pandas as pd
import os
import numpy as np
from load_data import load_data, split_pd, explore_data, plot_feature_pairs, plot_heatmap
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import RobustScaler

#%% 

def apply_winsorization(df, limits=(0.02, 0.02)):
    """
    Limits (0.01, 0.01) means:
    - Values below the 1st percentile are set to the 1st percentile.
    - Values above the 99th percentile are set to the 99th percentile.

    Parameters
    ----------
    pd.DataFrame
    
    Returns
    -------
    pd.DataFrame
        winsorized dataframe
    """
    df_winsorized = df.copy()
    
    for col in df_winsorized.columns:
        if np.issubdtype(df_winsorized[col].dtype, np.number):
            df_winsorized[col] = winsorize(df_winsorized[col], limits=limits)
            
    return df_winsorized

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
    return df_normalized

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
#%%

GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)

#%%

winsorized_GIST_train = apply_winsorization(GIST_train)
normalized_GIST_train = apply_normalization(winsorized_GIST_train)
preproc_GIST_train, kept_features = remove_zero_variance_features(normalized_GIST_train, show_details=False)
