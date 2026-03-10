import pandas as pd
import os
import numpy as np
from load_data import load_data, split_pd, explore_data, plot_feature_pairs, plot_heatmap
from preprocessing import apply_winsorization, apply_normalization, remove_zero_variance_features
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import RobustScaler
from skfeature.function.similarity_based import fisher_score
from scipy.stats import mannwhitneyu


#%% fisher feature selection

def fisher_feature_selection(X, y, k=20, show_details=True):
    X_np = X.to_numpy(dtype=float)
    y_np = np.asarray(y).ravel()

    scores = fisher_score.fisher_score(X_np, y_np)
    ranked_idx = np.argsort(scores)[::-1]

    top_idx = ranked_idx[:k]
    selected_features = X.columns[top_idx]
    fisher_selected_features = X.loc[:, selected_features]

    if show_details:
        print("Selected features:")
        print(selected_features.tolist())

    return fisher_selected_features, top_idx, scores


#%% Withney U-mann feature selection

def mann_whitney_u_feature_selection(X, y, k=20, show_details=True):

    X_np = X.to_numpy(dtype=float)
    y_np = np.asarray(y).ravel()

   
#%% 
GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, show_details = False)
winsorized_GIST_train = apply_winsorization(GIST_train)
normalized_GIST_train = apply_normalization(winsorized_GIST_train)
preproc_GIST_train, kept_features = remove_zero_variance_features(normalized_GIST_train, show_details=False)


#%% print selected features

# Fisher feature selection
fisher_selected_features, fisher_idx, fisher_scores = fisher_feature_selection(
    preproc_GIST_train, y_train, k=20, show_details=True
)

plot_heatmap(fisher_selected_features)

# Withney U-mann feature selection