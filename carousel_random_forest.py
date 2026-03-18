#%% Imports
import itertools
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from load_data import load_data, split_pd
from preprocessing import remove_zero_variance_features, remove_highly_correlated_features
from fs_lasso import fs_lasso
from fs_mRMR import fs_mrmr
from fs_mutualinformation import fs_mutualinformation
from fs_RFE import perform_rfe

#%% ---------------- SETTINGS ----------------

C_VALUES = [0.01, 0.02, 0.03]
K_VALUES = [10, 15, 20]
C_VALUES = [0.02]
K_VALUES = [15]

fs_configs = (
    [{'method': 'lasso', 'param': c} for c in C_VALUES] +
    [{'method': 'mrmr', 'param': k} for k in K_VALUES] +
    [{'method': 'mi', 'param': k} for k in K_VALUES] +
    [{'method': 'rfe', 'param': k} for k in K_VALUES]
)

rf_param_grid = {
    'n_estimators': [100, 200, 300],       
    'max_depth': [3, 5, 7],      
    'min_samples_split': [4, 6, 10],     
    'min_samples_leaf': [2, 5, 10],       
    'max_features': ['sqrt', 'log2', 0.3]  
}

rf_param_grid = {
    'n_estimators': [200],
    'max_depth': [5],
    'min_samples_split': [6],
    'min_samples_leaf': [3],
    'max_features': ['sqrt']
}
rf_keys, rf_values = zip(*rf_param_grid.items())
rf_param_combinations = [dict(zip(rf_keys, v)) for v in itertools.product(*rf_values)]


#%% ---------------- LOAD & PREPARE DATA ----------------

GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)

preproc_GIST_train, _ = remove_zero_variance_features(GIST_train, show_details=False)

X = preproc_GIST_train
y = y_train.values


#%% ---------------- NESTED CROSS-VALIDATION ----------------

# Outer loop evaluates model performance; Inner loop tunes hyperparameters
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

outer_results = []

for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    print(f"\n{'='*10} Outer Fold {outer_fold + 1} {'='*10}")

    # 1. Split Outer Data
    X_train_outer = X.iloc[train_idx]
    X_test_outer = X.iloc[test_idx]
    y_train_outer = y[train_idx]
    y_test_outer = y[test_idx]

    # 2. Scale Outer Data (Fit ONLY on training data to prevent leakage)
    scaler = RobustScaler()
    X_train_outer_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_outer),
        columns=X_train_outer.columns
    )
    X_test_outer_scaled = pd.DataFrame(
        scaler.transform(X_test_outer),
        columns=X_test_outer.columns
    )

    best_inner_score = -1
    best_fs_config = None
    best_rf_params = None

    # ================= INNER LOOP (Hyperparameter Tuning) =================
    for fs_config in fs_configs:
        for rf_params in rf_param_combinations:
            
            inner_scores = []

            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer_scaled, y_train_outer):
                
                # Split Inner Data
                X_train_inner = X_train_outer_scaled.iloc[inner_train_idx].copy()
                X_val_inner = X_train_outer_scaled.iloc[inner_val_idx].copy()
                y_train_inner = pd.Series(y_train_outer[inner_train_idx], index=X_train_inner.index)
                y_val_inner = pd.Series(y_train_outer[inner_val_idx], index=X_val_inner.index)

                # Remove Highly Correlated Features (Fit on inner train, apply to inner val)
                X_train_inner, kept_features = remove_highly_correlated_features(
                    X_train_inner,
                    correlation_threshold=0.95,
                    show_details=False
                )
                X_val_inner = X_val_inner[kept_features]

                # Execute Feature Selection
                if fs_config['method'] == 'lasso':
                    _, selected_features = fs_lasso(X_train_inner, y_train_inner, C=fs_config['param'])
                elif fs_config['method'] == 'mrmr':
                    _, selected_features = fs_mrmr(X_train_inner, y_train_inner, K=fs_config['param'], show_details=False)
                elif fs_config['method'] == 'mi':
                    selected_features, _ = fs_mutualinformation(X_train_inner, y_train_inner, k=fs_config['param'], showdetails=False)
                elif fs_config['method'] == 'rfe':
                    _, selected_features = perform_rfe(X_train_inner, y_train_inner, n_features_to_select=fs_config['param'])

                # Validate selected features
                selected_features = [f for f in selected_features if f in X_train_inner.columns]
                if not selected_features:
                    continue  # Skip if no features were selected

                # Subset data to selected features
                X_train_sel = X_train_inner[selected_features]
                X_val_sel = X_val_inner[selected_features]

                # Train Random Forest and evaluate
                rf = RandomForestClassifier(**rf_params, n_jobs=-1, random_state=42)
                rf.fit(X_train_sel, y_train_inner)
                
                probs = rf.predict_proba(X_val_sel)[:, 1]
                inner_scores.append(roc_auc_score(y_val_inner, probs))

            # Calculate average score for this parameter combination
            if not inner_scores:
                continue

            avg_score = np.mean(inner_scores)

            # Update best configuration if current is better
            if avg_score > best_inner_score:
                best_inner_score = avg_score
                best_fs_config = fs_config
                best_rf_params = rf_params

    print(f"Best Configuration -> FS: {best_fs_config}, RF: {best_rf_params}")

    # ================= OUTER EVALUATION (Model Validation) =================
    
    # Process outer training data with correlation removal
    X_train_outer_corr, kept_features_outer = remove_highly_correlated_features(
        X_train_outer_scaled,
        correlation_threshold=0.95,
        show_details=False
    )
    X_test_outer_corr = X_test_outer_scaled[kept_features_outer]
    y_train_outer_series = pd.Series(y_train_outer, index=X_train_outer_corr.index)

    # Apply the BEST Feature Selection method found in the inner loop
    if best_fs_config['method'] == 'lasso':
        _, final_features = fs_lasso(X_train_outer_corr, y_train_outer_series, C=best_fs_config['param'])
    elif best_fs_config['method'] == 'mrmr':
        _, final_features = fs_mrmr(X_train_outer_corr, y_train_outer_series, K=best_fs_config['param'], show_details=False)
    elif best_fs_config['method'] == 'mi':
        final_features, _ = fs_mutualinformation(X_train_outer_corr, y_train_outer_series, k=best_fs_config['param'], showdetails=False)
    elif best_fs_config['method'] == 'rfe':
        _, final_features = perform_rfe(X_train_outer_corr, y_train_outer_series, n_features_to_select=best_fs_config['param'])

    # Validate final features
    final_features = [f for f in final_features if f in X_train_outer_corr.columns]

    # Train final outer model and evaluate on the true holdout set (outer test)
    if not final_features:
        outer_score = 0
    else:
        final_rf = RandomForestClassifier(**best_rf_params, n_jobs=-1, random_state=42)
        final_rf.fit(X_train_outer_corr[final_features], y_train_outer)
        
        probs = final_rf.predict_proba(X_test_outer_corr[final_features])[:, 1]
        outer_score = roc_auc_score(y_test_outer, probs)

    # Store consolidated results for this fold
    outer_results.append({
        'fold': outer_fold + 1,
        'model_name': f"{best_fs_config['method']}_RF",
        'best_fs_method': best_fs_config['method'],
        'best_fs_param': best_fs_config['param'],
        'best_rf_params': best_rf_params,
        'n_features_selected': len(final_features),
        'roc_auc_score': outer_score
    })


# ---------------- SAVE & DISPLAY RESULTS ----------------

results_df = pd.DataFrame(outer_results)

print("\n" + "="*20 + " RESULTS " + "="*20)
print(results_df.to_string(index=False))

# Calculate Valid Scores
valid_scores = results_df['roc_auc_score'].dropna()
valid_scores = valid_scores[valid_scores.apply(lambda x: isinstance(x, (int, float)))]

if not valid_scores.empty:
    print(f"\nAverage Test ROC AUC Score: {valid_scores.mean():.3f} +/- {valid_scores.std():.3f}")

# 1. Save to CSV
results_df.to_csv('nested_cv_results_RF.csv', index=False)

# 2. Extract scores for Wilcoxon testing and save to Pickle
all_model_scores = {}

for _, row in results_df.iterrows():
    if pd.isna(row.get('model_name')):
        continue

    model_name = row['model_name']
    score = row['roc_auc_score']

    if not isinstance(score, (int, float)):
        continue

    if model_name not in all_model_scores:
        all_model_scores[model_name] = []
    
    all_model_scores[model_name].append(score)

with open('model_scores_RF.pkl', 'wb') as f:
    pickle.dump(all_model_scores, f)

print(f"\nScores per model (Saved for Wilcoxon):\n{all_model_scores}")