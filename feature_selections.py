#%%

import pandas as pd
import numpy as np
from load_data import load_data, split_pd, explore_data, plot_feature_pairs, plot_heatmap
from preprocessing import apply_winsorization, apply_normalization
from sklearn.linear_model import LogisticRegressionCV

#%%

def reduce_features(df, correlation_threshold=0.97, show_details = True):
    """
    removes features if:
        - variance  = 0 (all values are the same)
        - correlation > correlation_threshold (feature is very similar to another one) 
    Parameters
    ----------
    pd.DataFrame
    
    Returns
    -------
    pd.DataFrame
        reduced features dataframe

    Index
        features indexes

    """
    zero_var_cols = df.columns[~(df != df.iloc[0]).any()]
    df_reduced_var = df.drop(columns=zero_var_cols)

    corr_matrix = df_reduced_var.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    corr_drop_cols = [
        column for column in upper.columns
        if any(upper[column] > correlation_threshold)
    ]

    df_reduced = df_reduced_var.drop(columns=corr_drop_cols)

    kept_features = df_reduced.columns

    if show_details:
        print(f"Number of features before: {df.shape[1]}")
        print(f"Dropped zero variance features: {len(zero_var_cols)}")
        print(f"Dropped highly correlated features: {len(corr_drop_cols)}")
        print(f"Remaining features: {len(kept_features)}")

    return df_reduced, kept_features

def lasso_feature_selection(df, y, show_details = True):
    """
    LASSO-style feature selection using Logistic Regression with L1 penalty.
    
    Parameters
    ----------
    pd.DataFrame
    
    Returns
    -------
    pd.DataFrame
        reduced features dataframe

    index
        features indexes
    """

    model = LogisticRegressionCV(
        penalty='l1',
        solver='liblinear',
        cv=5,
        Cs=[0.1, 1, 5, 10, 20, 50],
        max_iter=5000,
        random_state=42
    )

    model.fit(df, y)

    coefs = model.coef_[0]
    mask = coefs != 0

    selected_features = df.columns[mask]
    if show_details:
        print("Selected features:", len(selected_features))
        print(selected_features)

    X_selected = df.loc[:, selected_features]

    return X_selected, selected_features

#%%

GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, show_details = False)
winsorized_GIST_train = apply_winsorization(GIST_train)
normalized_GIST_train = apply_normalization(winsorized_GIST_train)

#%%

filtered_GIST_train, corr_features_index = reduce_features(normalized_GIST_train,correlation_threshold=0.97, show_details=False)
feature_selected_GIST_train, features_index = lasso_feature_selection(filtered_GIST_train,y_train, show_details=False)
##%

# hallo
