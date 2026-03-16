import pandas as pd
import os
import numpy as np
from load_data import load_data, split_pd, explore_data, plot_feature_pairs, plot_heatmap
from preprocessing import apply_normalization, remove_zero_variance_features
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import RobustScaler
from skfeature.function.similarity_based import fisher_score
from scipy.stats import mannwhitneyu


#%% fisher feature selection

def remove_highly_correlated_features(df, correlation_threshold=0.99, show_details=True):
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

def fs_statistical(df, y_train, k=20):
    df_red, _ = remove_highly_correlated_features(
        df,
        correlation_threshold=0.9,
        show_details=False
    )

    scores = fisher_score.fisher_score(
        df_red.values.astype(float),
        np.asarray(y_train).ravel()
    )
    idx = np.argsort(scores)[::-1][:k]
    selected_features = df_red.columns[idx]

    return df_red[selected_features], selected_features
   
#%% 
GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)
normalized_GIST_train, scaler = apply_normalization(GIST_train)
preproc_GIST_train, kept_features = remove_zero_variance_features(normalized_GIST_train, show_details=False)

#%% Fisher feature selection
GIST_train_fisher, selected_features = fs_statistical(preproc_GIST_train, y_train)

print(selected_features)
plot_heatmap(GIST_train_fisher)


#%% Withney U-mann feature selection

def mann_whitney_u_feature_selection(X, y, k=20, show_details=True):

    X_np = X.to_numpy(dtype=float)
    y_np = np.asarray(y).ravel()

# Withney U-mann feature selection