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

#%%

GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)

#%%
winsorized_GIST_train = apply_winsorization(GIST_train)
normalized_GIST_train = apply_normalization(winsorized_GIST_train)
from load_data import load_data, split_pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)


##% Oproepen data en testset weer verkrijgen

GIST_data = load_data("GIST_radiomicFeatures.csv")

GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data)

#%% Variance controleren, if there are features with zero variance, remove them?

zero_variance_features = GIST_train.columns[GIST_train.var() == 0]

print(f"Features with zero variance: {list(zero_variance_features)}")
print("Number of zero-varinace features:", len(zero_variance_features))

#%% Almost zero variance features (threshold = 0.01)

variance = GIST_train.var()

threshold = 0.001

low_variance_features = variance[variance < threshold].index

print(f"Features with variance below {threshold}: {list(low_variance_features)}")
print("Number of low-variance features:", len(low_variance_features))


#%% Missing values 

def missing_values(df):
    
    print("\nMissing Values (Top 10):")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing.head(10) if not missing.empty else "No missing values found.")

missing_values(GIST_train)

#%% Outliers detecteren met IQR-methode

def list_outliers_iqr(X, y, multiplier=1.5, max_per_feature=None):
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR

    outlier_mask = (X.lt(lower) | X.gt(upper))  # DataFrame met True/False

    records = []

    for feature in X.columns:
        idxs = outlier_mask.index[outlier_mask[feature]]
        if max_per_feature is not None:
            idxs = idxs[:max_per_feature]
        for idx in idxs:
            val = X.at[idx, feature]
            records.append({
                "index": idx,
                "label": y[idx],
                "feature": feature,
                "value": X.loc[idx, feature],
                "lower_bound": lower[feature],
                "upper_bound": upper[feature],
            })

    return pd.DataFrame(records)

outliers_iqr = list_outliers_iqr(GIST_train, y_train, multiplier=1.5)

outliers_per_feature_label = (
    outliers_iqr
    .groupby(["feature", "label"])
    .size()
    .unstack(fill_value=0)
)

print(outliers_per_feature_label)

#%% Overlap tussen outliers en low-variance features

def find_overlap_outliers_low_variance(outliers_df, low_variance_features):
    outliers_features = set(outliers_df['feature'])
    low_variance_set = set(low_variance_features)
    overlap = outliers_features.intersection(low_variance_set)

    print(f"Overlap between outliers and low-variance features: {overlap}")
    print("Number of overlap between outliers and low-variance features:", len(overlap))

    return overlap

overlap_low_variance = find_overlap_outliers_low_variance(outliers_iqr, low_variance_features)

#%% Overlap tussen outliers en low-variance features

def find_overlap_outliers_zero_variance(outliers_df, zero_variance_features):
    outliers_features = set(outliers_df['feature'])
    zero_variance_set = set(zero_variance_features)
    overlap = outliers_features.intersection(zero_variance_set)

    print(f"Overlap between outliers and zero-variance features: {overlap}")
    print("Number of overlap between outliers and zero-variance features:", len(overlap))

    return overlap

overlap_zero = find_overlap_outliers_zero_variance(outliers_iqr, zero_variance_features)

#%% Normalisation with z-score

def normalise_zscore(X):
    X_normalised_zscore = X.copy()

    for feature in X.columns:
        mean = X[feature].mean()
        std = X[feature].std()
        if std != 0:
            X_normalised_zscore[feature] = (X[feature] - mean) / std
        else:
            X_normalised_zscore[feature] = 0  

    return X_normalised_zscore

X_normalized_zscore = normalise_zscore(GIST_train)

#%% Normalisation with IQR

def normalise_iqr(X):
    X_normalised_iqr = X.copy()

    for feature in X.columns:
        Q1 = X[feature].quantile(0.25)
        Q3 = X[feature].quantile(0.75)
        IQR = Q3 - Q1
        if IQR != 0:
            X_normalised_iqr[feature] = (X[feature] - Q1) / IQR
        else:
            X_normalised_iqr[feature] = 0

    return X_normalised_iqr

X_normalised_iqr = normalise_iqr(GIST_train)

print(GIST_train.describe())
print(X_normalised_iqr.describe())











# %%
