#%% 

import pandas as pd
import numpy as np
from load_data import load_data, split_pd, explore_data, plot_feature_pairs, plot_heatmap
from preprocessing import apply_normalization, remove_zero_variance_features
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
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
        penalty="l1",
        solver="saga",
        Cs=50,
        cv=5,
        scoring="roc_auc",
        max_iter=10000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        show_details = False):
    """
    Perform feature selection using regularized logistic regression (Elastic Net / LASSO).

    Parameters
    ----------
    df : pd.DataFrame
        Input features.
    y : array-like
        Target labels.
    penalty : str, default="saga"
        Type of regularization. 'elasticnet' is recommended for correlated features.
        Set to 'l1' for pure LASSO.
    l1_ratios : list of floats
        The Elastic-Net mixing parameter. 1.0 is pure L1 (LASSO), 0.0 is pure L2 (Ridge).
    Cs : int or array
        Regularization strengths tested during CV.
    """

    # Configure LogisticRegressionCV based on penalty type
    model = LogisticRegressionCV(
        penalty=penalty,
        solver=solver,
        Cs=Cs,
        cv=cv,
        scoring=scoring,
        max_iter=max_iter,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state
    )
    pipeline = Pipeline([("model", model)])
    
    pipeline.fit(df, y)
    fitted_model = pipeline.named_steps["model"]
    
    if fitted_model.coef_.ndim > 1 and fitted_model.coef_.shape[0] == 1:
        coefs = fitted_model.coef_[0]
    else:
        coefs = np.max(np.abs(fitted_model.coef_), axis=0)

    selected_features = list(df.columns[coefs != 0])
    df_selected = df[selected_features]

    if show_details:
        print(f"Features before selection: {df.shape[1]}")
        print(f"Selected features: {len(selected_features)}")
        
        if penalty == "elasticnet":
            print(f"Optimal l1_ratio chosen: {fitted_model.l1_ratio_[0]}")

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
    preproc_GIST_train_wo_high_corr_features, kept_features = remove_highly_correlated_features(
        df, correlation_threshold=0.97,
        show_details=False
    )
    GIST_train_lasso, kept_features = lasso_feature_selection(
        preproc_GIST_train_wo_high_corr_features,
        y_train,
        penalty="l1",
        solver="saga",
        Cs=np.logspace(-3, 3, 50),
        cv=5,
        scoring="roc_auc",
        max_iter=10000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        show_details = False
        )

    return GIST_train_lasso, kept_features

#%% DEZE PIPELINE kopieren

GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)
normalized_GIST_train, scaler = apply_normalization(GIST_train)
preproc_GIST_train, kept_features = remove_zero_variance_features(normalized_GIST_train, show_details=False)

#%%

GIST_train_lasso, kept_features = fs_lasso(preproc_GIST_train, y_train)