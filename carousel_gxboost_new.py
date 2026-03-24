#%% Imports

import itertools
import pickle
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score

from load_data import load_data, split_pd
from preprocessing import remove_zero_variance_features, remove_highly_correlated_features

from fs_lasso import fs_lasso
from fs_mRMR import fs_mrmr
from fs_mutualinformation import fs_mutualinformation
from fs_RFE import perform_rfe

#%% ---------------- SETTINGS ----------------

# Set your feature selection tuning grids here
C_VALUES = [0.01, 0.02, 0.03]
K_VALUES = [10, 15, 20]

C_VALUES = [0.02]
K_VALUES = [15]

# Build a list of all Feature Selection configurations to test
fs_configs = [{'method': 'lasso', 'param': c} for c in C_VALUES] + \
             [{'method': 'mrmr', 'param': k} for k in K_VALUES] + \
             [{'method': 'mi', 'param': k} for k in K_VALUES] + \
             [{'method': 'rfe', 'param': k} for k in K_VALUES]

# XGBoost specific hyperparameter grid (tailored for small datasets)
xgb_param_grid = {
    'n_estimators': [50, 100, 200],      # Number of trees
    'max_depth': [3, 4, 5],              # Keep trees shallow to prevent overfitting
    'learning_rate': [0.01, 0.05, 0.1],  # Step size shrinkage
    'subsample': [0.6, 0.8, 1.0],        # Fraction of samples used per tree
    'colsample_bytree': [0.6, 0.8, 1.0]  # Fraction of features used per tree
}

xgb_param_grid = {
   'n_estimators': [100],
   'max_depth': [4],     
   'learning_rate': [0.05],
   'subsample': [0.8],      
   'colsample_bytree': [0.8]
}

xgb_keys, xgb_values = zip(*xgb_param_grid.items())
xgb_param_combinations = [dict(zip(xgb_keys, v)) for v in itertools.product(*xgb_values)]

#%% ---------------- LOAD & PREPROCESS ----------------

GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)
preproc_GIST_train, _ = remove_zero_variance_features(GIST_train, show_details=False)

X = preproc_GIST_train 
y = y_train.values 

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#%% ---------------- NESTED CV ----------------

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)


outer_results = []

for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    print(f"\n--- Starting Outer Fold {outer_fold + 1} ---")
    
    X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]
    
    scaler = RobustScaler()
    X_train_outer_scaled = pd.DataFrame(scaler.fit_transform(X_train_outer), columns=X_train_outer.columns)
    X_test_outer_scaled = pd.DataFrame(scaler.transform(X_test_outer), columns=X_test_outer.columns)
    
    best_results = {
        'lasso': {'score': -1, 'fs_config': None, 'xgb_params': None},
        'mrmr':  {'score': -1, 'fs_config': None, 'xgb_params': None},
        'mi':    {'score': -1, 'fs_config': None, 'xgb_params': None},
        'rfe':   {'score': -1, 'fs_config': None, 'xgb_params': None}
    }

    # --- INNER LOOP: Hyperparameter Tuning ---
    for fs_config in fs_configs:
        for xgb_params in xgb_param_combinations:
            
            inner_scores = []
            
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer_scaled, y_train_outer):
                X_train_inner = X_train_outer_scaled.iloc[inner_train_idx]
                X_val_inner = X_train_outer_scaled.iloc[inner_val_idx]

                y_train_inner = pd.Series(y_train_outer[inner_train_idx], index=X_train_inner.index)
                y_val_inner = pd.Series(y_train_outer[inner_val_idx], index=X_val_inner.index)

                X_train_inner, kept_features = remove_highly_correlated_features(
                    X_train_inner,
                    correlation_threshold=0.95,
                    show_details=False
                )

                X_val_inner = X_val_inner[kept_features]

                # 1. Apply Feature Selection
                if fs_config['method'] == 'lasso':
                    _, selected_features = fs_lasso(X_train_inner, y_train_inner, C=fs_config['param'])
                elif fs_config['method'] == 'mrmr':
                    _, selected_features = fs_mrmr(X_train_inner, y_train_inner, K=fs_config['param'], show_details=False)
                elif fs_config['method'] == 'mi':
                    selected_features, _ = fs_mutualinformation(X_train_inner, y_train_inner, k=fs_config['param'], showdetails=False)
                elif fs_config['method'] == 'rfe':
                    _, selected_features = perform_rfe(X_train_inner, y_train_inner, n_features_to_select=fs_config['param'])

                selected_features = list(selected_features)
                selected_features = [f for f in selected_features if f in X_train_inner.columns]
                
                if not selected_features:
                    continue 
                
                X_train_inner_sel = X_train_inner[selected_features]
                X_val_inner_sel = X_val_inner[selected_features]
                
                # 2. Train XGBoost
                xgb_model = XGBClassifier(**xgb_params, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
                xgb_model.fit(X_train_inner_sel, y_train_inner)
                
                probs = xgb_model.predict_proba(X_val_inner_sel)[:, 1]
                inner_scores.append(roc_auc_score(y_val_inner, probs))      

            if not inner_scores:
                continue
                

            avg_score = np.mean(inner_scores)
            method = fs_config['method']

            # Update the dictionary if the current score beats the saved best score for that method
            if avg_score > best_results[method]['score']:
                best_results[method]['score'] = avg_score
                best_results[method]['fs_config'] = fs_config
                best_results[method]['xgb_params'] = xgb_params    
    
    for method, result in best_results.items():
        print(f"Best Configuration ({method.upper()}) -> FS: {result['fs_config']}, Score: {result['score']:.4f}, xgb Params: {result['xgb_params']}")

    # --- OUTER LOOP: Evaluate the Best Model Pipeline ---
    
        # --- Correlation filtering ---
    X_train_outer_corr, kept_features_outer = remove_highly_correlated_features(
        X_train_outer_scaled,
        correlation_threshold=0.95,
        show_details=False
    )

    X_test_outer_corr = X_test_outer_scaled[kept_features_outer]
    y_train_outer_series = pd.Series(y_train_outer, index=X_train_outer_corr.index)

    for method, result in best_results.items():
        best_fs_config = result['fs_config']
        best_xgb_params = result['xgb_params']

        # Safety check: skip if the method didn't find any valid config in the inner loop
        if best_fs_config is None or best_xgb_params is None:
            continue

        # Apply best FS
        if method == 'lasso':
            _, final_features = fs_lasso(X_train_outer_corr, y_train_outer_series, C=best_fs_config['param'])
        elif method == 'mrmr':
            _, final_features = fs_mrmr(X_train_outer_corr, y_train_outer_series, K=best_fs_config['param'], show_details=False)
        elif method == 'mi':
            final_features, _ = fs_mutualinformation(X_train_outer_corr, y_train_outer_series, k=best_fs_config['param'], showdetails=False)
        elif method == 'rfe':
            _, final_features = perform_rfe(X_train_outer_corr, y_train_outer_series, n_features_to_select=best_fs_config['param'])

        final_features = [f for f in final_features if f in X_train_outer_corr.columns]

        if not final_features:
            outer_score = 0
        else:
            final_xgb = XGBClassifier(**best_xgb_params, n_jobs=-1, eval_metric='logloss')
            final_xgb.fit(X_train_outer_corr[final_features], y_train_outer)

            probs = final_xgb.predict_proba(X_test_outer_corr[final_features])[:, 1]
            outer_score_auc = roc_auc_score(y_test_outer, probs)
            preds = (probs >= 0.5).astype(int)
            outer_score_acc = accuracy_score(y_test_outer, preds)

        outer_results.append({
            'fold': outer_fold + 1,
            'model_name': f"{method.upper()}_XGB", 
            'fs_method': method,
            'best_fs_param': best_fs_config['param'],
            'best_xgb_params': best_xgb_params,
            'n_features_selected': len(final_features),
            'roc_auc_score': outer_score_auc,
            'accuracy_score': outer_score_acc

        }) 

#---------------- FINAL RESULTS ----------------
results_df = pd.DataFrame(outer_results)

print("\n" + "="*20 + " RESULTS " + "="*20)
print(results_df.to_string(index=False))

# Calculate and print the average and std per feature selection method
if not results_df.empty:
    print("\nAverage Test ROC AUC Score per Model:")
    summary_stats = results_df.groupby('model_name')['roc_auc_score'].agg(['mean', 'std'])
    for index, row in summary_stats.iterrows():
        print(f"{index}: {row['mean']:.3f} +/- {row['std']:.3f}")
    print("\nAverage Test Accuracy Score per Model:")
    summary_stats = results_df.groupby('model_name')['accuracy_score'].agg(['mean', 'std'])
    for index, row in summary_stats.iterrows():
        print(f"{index}: {row['mean']:.3f} +/- {row['std']:.3f}")



# Save CSV
results_df.to_csv('nested_cv_results_XGB.csv', index=False)

# Save pickle (Wilcoxon)
all_model_scores = {}

for _, row in results_df.iterrows():
    model_name = row['model_name']
    score = row['roc_auc_score']

    if model_name not in all_model_scores:
        all_model_scores[model_name] = []

    all_model_scores[model_name].append(score)

with open('model_scores_XGB.pkl', 'wb') as f:
    pickle.dump(all_model_scores, f)

print("\nScores per model:")
for model, scores in all_model_scores.items():
    print(f"{model}: {[f'{s:.4f}' for s in scores]}")

print("\n=== Processing Complete ===")