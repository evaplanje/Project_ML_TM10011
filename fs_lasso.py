
import pandas as pd
import numpy as np
from load_data import load_data, split_pd, explore_data, plot_feature_pairs, plot_heatmap
from preprocessing import apply_winsorization, apply_normalization
from sklearn.linear_model import LogisticRegressionCV

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

    df_reduced = df.columns[mask]
    if show_details:
        print("Selected features:", len(df_reduced))
        print(df_reduced)

    kept_features = df.loc[:, df_reduced]

    return df_reduced, kept_features


GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, show_details = False)
winsorized_GIST_train = apply_winsorization(GIST_train)
normalized_GIST_train = apply_normalization(winsorized_GIST_train)

#%%

filtered_GIST_train, corr_features_index = reduce_features(normalized_GIST_train,correlation_threshold=0.97, show_details=False)
feature_selected_GIST_train, features_index = lasso_feature_selection(filtered_GIST_train,y_train, show_details=False)
##%
