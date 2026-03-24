#%% Imports

import itertools
import pickle
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

from load_data import load_data, split_pd
from preprocessing import remove_zero_variance_features, remove_highly_correlated_features

from fs_lasso import fs_lasso
from fs_mRMR import fs_mrmr
from fs_mutualinformation import fs_mutualinformation
from fs_RFE import perform_rfe

#%% ---------------- SETTINGS ----------------

# Feature Selection params
C_VALUES = [0.01, 0.02, 0.03]
K_VALUES = [10, 15, 20]

fs_configs = (
    [{'method': 'lasso', 'param': c} for c in C_VALUES] +
    [{'method': 'mrmr', 'param': k} for k in K_VALUES] +
    [{'method': 'mi', 'param': k} for k in K_VALUES] +
    [{'method': 'rfe', 'param': k} for k in K_VALUES]
)

# ---------------- SVM PARAM GRID ----------------

#SVM_param_grid = {
 #   'C': [0.1, 1, 10],
  #  'kernel': ['linear', 'rbf', 'poly'],        #misschien leidt poly tot overfitting
   # 'gamma': ['scale', 0.01, 0.1]
#}

SVM_param_grid = {

    'C': [0.1, 1, 10],

    'kernel': ['linear', 'rbf'],

    'gamma': ['scale', 'auto']

}

# Create combinations
svm_keys, svm_values = zip(*SVM_param_grid.items())
svm_param_combinations = [dict(zip(svm_keys, v)) for v in itertools.product(*svm_values)]

#%% ---------------- LOAD & PREPARE DATA ----------------

GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)

preproc_GIST_train, _ = remove_zero_variance_features(GIST_train, show_details=False)

X = preproc_GIST_train
y = y_train.values

#%% ---------------- NESTED CROSS-VALIDATION ----------------

outer_cv = StratifiedKFold(n_splits=5, shuffle=True)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True)

outer_results = []

for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    print(f"\n{'='*10} Outer Fold {outer_fold + 1} {'='*10}")

    # 1. Split Outer Data
    X_train_outer = X.iloc[train_idx]
    X_test_outer = X.iloc[test_idx]
    y_train_outer = y[train_idx]
    y_test_outer = y[test_idx]

    # 2. Scale (NO leakage)
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
    best_svm_params = None

    # ================= INNER LOOP =================
    for fs_config in fs_configs:
        for svm_params in svm_param_combinations:

            inner_scores = []

            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer_scaled, y_train_outer):

                # Split inner
                X_train_inner = X_train_outer_scaled.iloc[inner_train_idx].copy()
                X_val_inner = X_train_outer_scaled.iloc[inner_val_idx].copy()

                y_train_inner = pd.Series(y_train_outer[inner_train_idx], index=X_train_inner.index)
                y_val_inner = pd.Series(y_train_outer[inner_val_idx], index=X_val_inner.index)

                # Correlation removal INSIDE CV
                X_train_inner, kept_features =
                remove_highly_correlated_features(
                    X_train_inner,
                    correlation_threshold=0.95,
                    show_details=False
                )
                X_val_inner = X_val_inner[kept_features]

                # Feature selection
                if fs_config['method'] == 'lasso':
                    _, selected_features = fs_lasso(X_train_inner, y_train_inner, C=fs_config['param'])
                elif fs_config['method'] == 'mrmr':
                    _, selected_features = fs_mrmr(X_train_inner, y_train_inner, K=fs_config['param'], show_details=False)
                elif fs_config['method'] == 'mi':
                    selected_features, _ = fs_mutualinformation(X_train_inner, y_train_inner, k=fs_config['param'], showdetails=False)
                elif fs_config['method'] == 'rfe':
                    _, selected_features = perform_rfe(X_train_inner, y_train_inner, n_features_to_select=fs_config['param'])

                selected_features = [f for f in selected_features if f in X_train_inner.columns]

                if not selected_features:
                    continue

                X_train_sel = X_train_inner[selected_features]
                X_val_sel = X_val_inner[selected_features]

                # Train SVM
                svm = SVC(**svm_params, probability=True)
                svm.fit(X_train_sel, y_train_inner)

                probs = svm.predict_proba(X_val_sel)[:, 1]
                inner_scores.append(roc_auc_score(y_val_inner, probs))

            if not inner_scores:
                continue

            avg_score = np.mean(inner_scores)

            if avg_score > best_inner_score:
                best_inner_score = avg_score
                best_fs_config = fs_config
                best_svm_params = svm_params

    print(f"Best Configuration -> FS: {best_fs_config}, SVM: {best_svm_params}")

    # ================= OUTER EVALUATION =================

    # Correlation removal on outer train
    X_train_outer_corr, kept_features_outer = remove_highly_correlated_features(
        X_train_outer_scaled,
        correlation_threshold=0.95,
        show_details=False
    )
    X_test_outer_corr = X_test_outer_scaled[kept_features_outer]

    y_train_outer_series = pd.Series(y_train_outer, index=X_train_outer_corr.index)

    # Apply best FS
    if best_fs_config['method'] == 'lasso':
        _, final_features = fs_lasso(X_train_outer_corr, y_train_outer_series, C=best_fs_config['param'])
    elif best_fs_config['method'] == 'mrmr':
        _, final_features = fs_mrmr(X_train_outer_corr, y_train_outer_series, K=best_fs_config['param'], show_details=False)
    elif best_fs_config['method'] == 'mi':
        final_features, _ = fs_mutualinformation(X_train_outer_corr, y_train_outer_series, k=best_fs_config['param'], showdetails=False)
    elif best_fs_config['method'] == 'rfe':
        _, final_features = perform_rfe(X_train_outer_corr, y_train_outer_series, n_features_to_select=best_fs_config['param'])

    final_features = [f for f in final_features if f in X_train_outer_corr.columns]

    if not final_features:
        outer_score = 0
    else:
        final_svm = SVC(**best_svm_params, probability=True)
        final_svm.fit(X_train_outer_corr[final_features], y_train_outer)

        probs = final_svm.predict_proba(X_test_outer_corr[final_features])[:, 1]
        outer_score = roc_auc_score(y_test_outer, probs)

    outer_results.append({
        'fold': outer_fold + 1,
        'model_name': f"{best_fs_config['method']}_SVM",
        'best_fs_method': best_fs_config['method'],
        'best_fs_param': best_fs_config['param'],
        'best_svm_params': best_svm_params,
        'n_features_selected': len(final_features),
        'roc_auc_score': outer_score
    })

# ---------------- RESULTS ----------------

results_df = pd.DataFrame(outer_results)

print("\n" + "="*20 + " RESULTS " + "="*20)
print(results_df.to_string(index=False))

valid_scores = results_df['roc_auc_score'].dropna()

if not valid_scores.empty:
    print(f"\nAverage Test ROC AUC Score: {valid_scores.mean():.3f} +/- {valid_scores.std():.3f}")

# Save CSV
results_df.to_csv('nested_cv_results_SVM.csv', index=False)

# Save pickle (Wilcoxon)
all_model_scores = {}

for _, row in results_df.iterrows():
    model_name = row['model_name']
    score = row['roc_auc_score']

    if model_name not in all_model_scores:
        all_model_scores[model_name] = []

    all_model_scores[model_name].append(score)

with open('model_scores_SVM.pkl', 'wb') as f:
    pickle.dump(all_model_scores, f)

print("\nScores per model:")
print(all_model_scores)

print("\n=== Processing Complete ===")

#%%