#%% 

import pandas as pd
import numpy as np
from load_data import load_data, split_pd, explore_data, plot_feature_pairs, plot_heatmap
from preprocessing import apply_normalization, remove_zero_variance_features
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


#%%
def remove_highly_correlated_features(df, correlation_threshold=0.97, show_details=True):
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
        penalty="elasticnet", 
        l1_ratios=[0.5, 0.7, 0.9, 1.0], 
        Cs=50,
        cv=5,
        scoring="roc_auc",
        solver="saga",
        max_iter=10000,
        class_weight="balanced", 
        apply_scaling=False, 
        show_details=True,
        random_state=42):
    """
    Perform feature selection using regularized logistic regression (Elastic Net / LASSO).

    Parameters
    ----------
    df : pd.DataFrame
        Input features.
    y : array-like
        Target labels.
    penalty : str, default="elasticnet"
        Type of regularization. 'elasticnet' is recommended for correlated features.
        Set to 'l1' for pure LASSO.
    l1_ratios : list of floats
        The Elastic-Net mixing parameter. 1.0 is pure L1 (LASSO), 0.0 is pure L2 (Ridge).
    Cs : int or array
        Regularization strengths tested during CV.
    apply_scaling : bool, default=False
        Whether to apply StandardScaler. Set to False if data is already standardized.
    ... [other parameters same as original]
    """

    steps = []
    if apply_scaling:
        steps.append(("scaler", StandardScaler()))
        
    # Configure LogisticRegressionCV based on penalty type
    if penalty == "elasticnet":
        model = LogisticRegressionCV(
            penalty=penalty,
            l1_ratios=l1_ratios,
            solver=solver,
            Cs=Cs,
            cv=cv,
            scoring=scoring,
            max_iter=max_iter,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=random_state
        )
    else:
        model = LogisticRegressionCV(
            penalty=penalty,
            solver=solver,
            Cs=Cs,
            cv=cv,
            scoring=scoring,
            max_iter=max_iter,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=random_state
        )

    steps.append(("model", model))
    pipeline = Pipeline(steps)
    
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
#%% 
GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, show_details = False)
normalized_GIST_train = apply_normalization(GIST_train)
preproc_GIST_train, kept_features = remove_zero_variance_features(normalized_GIST_train, show_details=False)

#%%

preproc_GIST_train_wo_high_corr_features, kept_features = remove_highly_correlated_features(
    preproc_GIST_train,
    show_details=True
)
#%%
preproc_GIST_train_lasso, kept_features = lasso_feature_selection(
    preproc_GIST_train_wo_high_corr_features,
    y_train,
    Cs=np.logspace(-5, 1, 50),  # test many regularization strengths
    cv=5,
    scoring="roc_auc",
    solver="saga",
    max_iter=10000,
    class_weight="balanced",
    show_details=True
)

#%%

plot_heatmap(preproc_GIST_train_lasso)

print(kept_features)