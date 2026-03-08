#%%

import pandas as pd
import os
import numpy as np
from load_data import load_data, split_pd, explore_data, plot_feature_pairs, plot_heatmap
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import RobustScaler


#%%
# Steps (https://www.sciencedirect.com/science/article/pii/S0933365721002232 + gemini)

# Clipping (remove outliers)
# Z-score normalization (https://www.sciencedirect.com/topics/computer-science/data-normalization)
# balacned feautre selection (mix shape + texture), (LASSO or Random Forest Feature Importance)
# multi-filter plot_feature_pairscost-senstive learning (weighted XGBoost/SVM)

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
        # We only winsorize numeric columns
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

#%%

GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data)


#%%
winsorized_GIST_train = apply_winsorization(GIST_train)
normalized_GIST_train = apply_normalization(winsorized_GIST_train)

