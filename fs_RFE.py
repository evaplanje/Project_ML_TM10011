
#%%
import pandas as pd
import numpy as np
from load_data import load_data, split_pd, explore_data, plot_feature_pairs, plot_heatmap
from preprocessing import apply_normalization, remove_zero_variance_features
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

# %%
def perform_rfe(X_train, y_train, n_features_to_select=20):
    """
    Perform true Recursive Feature Elimination (RFE) using Logistic Regression.
    """
    # 1. Define the base model (using standard LR to save computation time during RFE)
    base_model = LogisticRegression(solver = 'liblinear',max_iter=10000, random_state=7)
    
    # 2. Initialize RFE
    rfe = RFE(estimator=base_model, n_features_to_select=n_features_to_select, step=1)
    
    # 3. Fit RFE to the data (this runs the recursive loop)
    rfe.fit(X_train, y_train)
    
    # 4. Get the names of the selected features using the boolean mask (support_)
    selected_features = X_train.columns[rfe.support_].tolist()

    
    # Return a placeholder (or the dataframe) AND the list, so it unpacks correctly
    return None, selected_features


