#%% 

import pandas as pd
import numpy as np
from load_data import load_data, split_pd, explore_data, plot_feature_pairs, plot_heatmap
from preprocessing import apply_normalization, remove_zero_variance_features
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline


#%%
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

def lasso_feature_selection(
        df,
        y,
        C=0.1, 
        solver="saga",
        max_iter=10000,
        class_weight="balanced",
        # Removed n_jobs from the parameters here
        random_state=42,
        show_details=False):
    """
    Perform feature selection using standard regularized logistic regression (LASSO).
    Updated for scikit-learn >= 1.8 compatibility.
    """
    
    # Configure standard LogisticRegression without deprecated arguments
    model = LogisticRegression(
        l1_ratio=1,      # <-- This replaces penalty="l1" in newer scikit-learn versions
        C=C,          
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight,
        # <-- n_jobs has been completely removed
        random_state=random_state
    )
    
    pipeline = Pipeline([("model", model)])
    pipeline.fit(df, y)
    fitted_model = pipeline.named_steps["model"]
    
    if fitted_model.coef_.ndim > 1 and fitted_model.coef_.shape[0] == 1:
        coefs = fitted_model.coef_[0]
    else:
        coefs = np.max(np.abs(fitted_model.coef_), axis=0)

    # Prevent error if Lasso drops ALL features (coefs == 0)
    selected_features = list(df.columns[coefs != 0])
    
    # Fallback: If Lasso picks 0 features, grab the top 3 strongest features
    if len(selected_features) == 0:
        if show_details:
            print("Lasso dropped all features! Falling back to top 3 features.")
        top_indices = np.argsort(np.abs(coefs))[-3:]
        selected_features = list(df.columns[top_indices])

    df_selected = df[selected_features]

    if show_details:
        print(f"Features before selection: {df.shape[1]}")
        print(f"Selected features: {len(selected_features)}")
        
        importance = pd.DataFrame({
            "feature": df.columns,
            "coef": coefs
        })

        importance = importance[importance.coef != 0]
        importance["abs_coef"] = importance.coef.abs()
        importance = importance.sort_values("abs_coef", ascending=False)

        print("\nTop selected features:")
        print(importance[["feature", "coef"]].head(10))

    return df_selected, selected_features

def fs_lasso(df, y_train):
    preproc_GIST_train_wo_high_corr, kept_features = remove_highly_correlated_features(
        df, 
        correlation_threshold=0.97, 
        show_details=False
    )
    
    GIST_train_lasso, final_kept_features = lasso_feature_selection(
        preproc_GIST_train_wo_high_corr,
        y_train,
        C=0.01, 
        solver="saga",
        max_iter=10000,
        class_weight="balanced",
        # Make sure n_jobs=-1 is deleted here too!
        random_state=42,
        show_details=False
    )

    return GIST_train_lasso, final_kept_features


#%% DEZE PIPELINE kopieren

GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)
normalized_GIST_train, scaler = apply_normalization(GIST_train)
preproc_GIST_train, kept_features = remove_zero_variance_features(normalized_GIST_train, show_details=False)

#%%

GIST_train_lasso, kept_features = fs_lasso(preproc_GIST_train, y_train)