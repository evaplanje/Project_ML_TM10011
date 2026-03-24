
#%% 

import pandas as pd
import numpy as np
from load_data import load_data, split_pd, explore_data, plot_feature_pairs, plot_heatmap
from preprocessing import apply_normalization, remove_zero_variance_features
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline


#%%

def fs_lasso(
        df,
        y,
        C=0.1, 
        solver="saga",
        max_iter=10000,
        class_weight="balanced",
        show_details=False):
    """
    Perform feature selection using standard regularized logistic regression (LASSO).
    """
    
    # Configure standard LogisticRegression without deprecated arguments
    model = LogisticRegression(
        penalty='l1',   
        C=C,          
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=7
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


#%% DEZE PIPELINE kopieren

# GIST_data = load_data('GIST_radiomicFeatures.csv')
# GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)
# normalized_GIST_train, scaler = apply_normalization(GIST_train)
# preproc_GIST_train, kept_features = remove_zero_variance_features(normalized_GIST_train, show_details=False)

# #%%

# GIST_train_lasso, kept_features = fs_lasso(preproc_GIST_train.iloc[:106], y_train.iloc[:106], 0.05)