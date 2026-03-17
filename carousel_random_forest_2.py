#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
import itertools
import random

from fs_lasso import fs_lasso
from fs_mRMR import fs_mrmr
from load_data import load_data, split_pd
from preprocessing import remove_highly_correlated_features, remove_zero_variance_features

#%% ---------------- SETTINGS ----------------

fs_methods = {
    'lasso': fs_lasso,
    'mrmr': fs_mrmr
}

# LASSO grid
C_values = np.linspace(0.01, 0.1, 5)
k_values = [10, 20, 30, 40, 50]

# RF grid
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

rf_keys, rf_values = zip(*rf_param_grid.items())
all_rf_combinations = [dict(zip(rf_keys, v)) for v in itertools.product(*rf_values)]

# Sample only 10 RF configs to speed up tuning
rf_param_combinations = random.sample(all_rf_combinations, 10)

# Fixed RF for FS tuning (Stage 1)
rf_fixed = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    bootstrap=True,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)

#%% ---------------- LOAD & PREPROCESS ----------------

GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)

preproc_GIST_train, _ = remove_zero_variance_features(GIST_train, show_details=False)
preproc_GIST_train, _ = remove_highly_correlated_features(
    preproc_GIST_train, correlation_threshold=0.90, show_details=False
)

X = preproc_GIST_train
y = y_train.values

#%% ---------------- NESTED CV ----------------

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

outer_results = []

for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    print(f"\n--- Outer Fold {outer_fold + 1} ---")

    X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]

    # Scaling (resets index to 0...N)
    scaler = RobustScaler()
    X_train_outer_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_outer),
        columns=X_train_outer.columns
    )
    X_test_outer_scaled = pd.DataFrame(
        scaler.transform(X_test_outer),
        columns=X_test_outer.columns
    )

    # ==========================================================
    # 🔹 STAGE 1: TUNE LASSO and mRMR
    # ==========================================================
    best_fs_method = None
    best_fs_param = None
    best_fs_score = -1

    # ---------- LASSO ----------
    for C in C_values:
        inner_scores = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer_scaled, y_train_outer):
            X_train_inner = X_train_outer_scaled.iloc[inner_train_idx]
            y_train_inner = y_train_outer[inner_train_idx]
            X_val_inner = X_train_outer_scaled.iloc[inner_val_idx]
            y_val_inner = y_train_outer[inner_val_idx]

            _, selected_features = fs_lasso(X_train_inner, y_train_inner, C=C)
            selected_features = list(selected_features) # Safeguard

            if not selected_features:
                continue

            rf_fixed.fit(X_train_inner[selected_features], y_train_inner)
            preds = rf_fixed.predict(X_val_inner[selected_features])
            inner_scores.append(accuracy_score(y_val_inner, preds))

        if inner_scores and np.mean(inner_scores) > best_fs_score:
            best_fs_score = np.mean(inner_scores)
            best_fs_method = "lasso"
            best_fs_param = C

    # ---------- mRMR ----------
    for k in k_values:
        inner_scores = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer_scaled, y_train_outer):
            X_train_inner = X_train_outer_scaled.iloc[inner_train_idx]
            y_train_inner = pd.Series(y_train_outer[inner_train_idx], index=X_train_inner.index)
            X_val_inner = X_train_outer_scaled.iloc[inner_val_idx]
            y_val_inner = pd.Series(y_train_outer[inner_val_idx], index=X_val_inner.index)

            _, selected_features = fs_mrmr(X_train_inner, y_train_inner, K=k)
            
            # Force proper list and validate against columns
            selected_features = [f for f in list(selected_features) if f in X_train_inner.columns]

            if not selected_features:
                continue

            rf_fixed.fit(X_train_inner[selected_features], y_train_inner)
            preds = rf_fixed.predict(X_val_inner[selected_features])
            inner_scores.append(accuracy_score(y_val_inner, preds))

        if inner_scores and np.mean(inner_scores) > best_fs_score:
            best_fs_score = np.mean(inner_scores)
            best_fs_method = "mrmr"
            best_fs_param = k

    print(f"Best FS: {best_fs_method}, Param: {best_fs_param}")

    # ==========================================================
    # 🔹 STAGE 2: SELECT FEATURES WITH BEST PARAMETER
    # ==========================================================
    if best_fs_method == "lasso":
        _, selected_features = fs_lasso(X_train_outer_scaled, y_train_outer, C=best_fs_param)
    elif best_fs_method == "mrmr":
        _, selected_features = fs_mrmr(X_train_outer_scaled, y_train_outer, K=best_fs_param)
    else:
        print("Feature selection failed to find any valid features. Skipping fold.")
        continue

    # 🚨 THE FIX: Force selected_features to be a clean Python list of strings
    if hasattr(selected_features, 'tolist'):
        selected_features = selected_features.tolist()
    elif isinstance(selected_features, pd.DataFrame):
        selected_features = selected_features.columns.tolist()
    else:
        selected_features = list(selected_features)
        
    # Ensure they actually exist in the DataFrame
    selected_features = [f for f in selected_features if f in X_train_outer_scaled.columns]

    print(f"Selected {len(selected_features)} features")
    
    # Check if we accidentally zeroed out all features
    if len(selected_features) == 0:
        print("0 features selected. Skipping to next fold.")
        continue

    # ==========================================================
    # 🔹 STAGE 3: TUNE RF (on selected features only)
    # ==========================================================
    best_rf_params = None
    best_rf_score = -1

    for rf_params in rf_param_combinations:
        inner_scores = []

        for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer_scaled, y_train_outer):
            
            X_train_inner = X_train_outer_scaled.loc[inner_train_idx, selected_features]
            y_train_inner = y_train_outer[inner_train_idx]
            
            X_val_inner = X_train_outer_scaled.loc[inner_val_idx, selected_features]
            y_val_inner = y_train_outer[inner_val_idx]

            rf = RandomForestClassifier(
                **rf_params,
                bootstrap=True,
                max_features='sqrt',
                n_jobs=-1
            )

            rf.fit(X_train_inner, y_train_inner)
            preds = rf.predict(X_val_inner)
            inner_scores.append(accuracy_score(y_val_inner, preds))

        avg_score = np.mean(inner_scores)

        if avg_score > best_rf_score:
            best_rf_score = avg_score
            best_rf_params = rf_params

    print(f"Best RF params: {best_rf_params}")

    # ==========================================================
    # 🔹 FINAL MODEL (OUTER TEST)
    # ==========================================================
    final_rf = RandomForestClassifier(
        **best_rf_params,
        bootstrap=True,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    final_rf.fit(X_train_outer_scaled[selected_features], y_train_outer)
    final_preds = final_rf.predict(X_test_outer_scaled[selected_features])
    outer_score = accuracy_score(y_test_outer, final_preds)

    outer_results.append({
        'fold': outer_fold + 1,
        'best_fs_method': best_fs_method,
        'best_fs_param': best_fs_param,
        'n_features': len(selected_features),
        'best_rf_params': best_rf_params,
        'test_accuracy': outer_score
    })

#%% ---------------- FINAL RESULTS ----------------

results_df = pd.DataFrame(outer_results)

print("\n=== Final Results ===")
print(results_df)

if not results_df.empty:
    print(f"\nAverage Accuracy: "
          f"{results_df['test_accuracy'].mean():.3f} "
          f"+/- {results_df['test_accuracy'].std():.3f}")