#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import RobustScaler
import itertools
from fs_lasso import fs_lasso
from fs_mRMR import fs_mrmr
# from fs_statistical import fs_statistical
# from fs_RFE import perform_rfe
# from fs_groupwise import fs_groupwise
# from fs_pca import fs_pca

from load_data import load_data, split_pd, explore_data, plot_feature_pairs, plot_heatmap
from preprocessing import remove_zero_variance_features, remove_highly_correlated_features
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline


#%% ---------------- SETTINGS ----------------

# Set your fixed hyperparameters here
C_VALUES = [0.03, 0.06, 0.08]
K_VALUES = [10, 20, 30]

# Build a list of all Feature Selection configurations we want to test
fs_configs = [{'method': 'lasso', 'param': c} for c in C_VALUES] + \
             [{'method': 'mrmr', 'param': k} for k in K_VALUES]


rf_param_grid = {
    'n_estimators': [100],
    'max_depth': [5],
    'min_samples_split': [2]
}

# Create a list of all RF parameter combinations
rf_keys, rf_values = zip(*rf_param_grid.items())
rf_param_combinations = [dict(zip(rf_keys, v)) for v in itertools.product(*rf_values)]

#%% ---------------- LOAD & PREPROCESS ----------------

# import
GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)
preproc_GIST_train, kept_features = remove_zero_variance_features(GIST_train, show_details=False)
preproc_GIST_train, kept_features = remove_highly_correlated_features(preproc_GIST_train, correlation_threshold=0.90, show_details=False)

X = preproc_GIST_train # Your pandas DataFrame (200, n_features)
y = y_train.values # Your classes array (200,)

#%% ---------------- NESTED CV ----------------

# Outer loop evaluates the model's true performance
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Inner loop tunes the hyperparameters and feature selection
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

outer_results = []

for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    print(f"\n--- Starting Outer Fold {outer_fold + 1} ---")
    
    # Split data for the outer loop
    X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]
    
    # Radiomics usually requires scaling (do this AFTER splitting to avoid leakage)
    scaler = RobustScaler()
    X_train_outer_scaled = pd.DataFrame(scaler.fit_transform(X_train_outer), columns=X_train_outer.columns)
    X_test_outer_scaled = pd.DataFrame(scaler.transform(X_test_outer), columns=X_test_outer.columns)

    best_inner_score = -1
    best_fs_method = None
    best_rf_params = None
    best_selected_features = None

    # --- INNER LOOP: Hyperparameter Tuning ---
    for fs_name, fs_func in fs_methods.items():
        for rf_params in rf_param_combinations:
            
            inner_scores = []
            
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer_scaled, y_train_outer):
                X_train_inner = X_train_outer_scaled.iloc[inner_train_idx]
                X_val_inner = X_train_outer_scaled.iloc[inner_val_idx]

                # 🚨 THE FIX: Force y into a Pandas Series and manually align the indices! 
                y_train_inner = pd.Series(y_train_outer[inner_train_idx], index=X_train_inner.index)
                y_val_inner = pd.Series(y_train_outer[inner_val_idx], index=X_val_inner.index)
                
                # 1. Apply Feature Selection on the Inner Training set
                _, selected_features = fs_func(X_train_inner, y_train_inner)
                
                # Safety check: Force list format and ensure features exist
                selected_features = list(selected_features)
                selected_features = [f for f in selected_features if f in X_train_inner.columns]
                
                if not selected_features:
                    continue # Skip if no features were selected
                
                # 2. Filter validation set to match selected features
                X_train_inner_sel = X_train_inner[selected_features]
                X_val_inner_sel = X_val_inner[selected_features]
                
                # 3. Train Random Forest
                rf = RandomForestClassifier(**rf_params, bootstrap=True, max_features='sqrt', random_state=42, n_jobs=-1)
                rf.fit(X_train_inner_sel, y_train_inner)
                
                # 4. Evaluate
                preds = rf.predict(X_val_inner_sel)
                score = accuracy_score(y_val_inner, preds)
                inner_scores.append(score)
            
            if not inner_scores:
                continue
                
            # Average score across inner folds for this combination
            avg_inner_score = np.mean(inner_scores)
            
            # Keep track of the best combination
            if avg_inner_score > best_inner_score:
                best_inner_score = avg_inner_score
                best_fs_method = fs_name
                best_rf_params = rf_params

    print(f"Best Inner Tuning - FS: {best_fs_method}, RF Params: {best_rf_params}")

    # --- OUTER LOOP: Evaluate the Best Model Pipeline ---
    winning_fs_func = fs_methods[best_fs_method]
    
    # 🚨 THE FIX AGAIN: Create a properly indexed Series for the outer loop feature selection!
    y_train_outer_series = pd.Series(y_train_outer, index=X_train_outer_scaled.index)
    
    # Re-apply the winning feature selection to the ENTIRE outer training set
    _, final_selected_features = winning_fs_func(X_train_outer_scaled, y_train_outer_series)
    
    # Enforce safe feature extraction for the outer loop
    final_selected_features = list(final_selected_features)
    final_selected_features = [f for f in final_selected_features if f in X_train_outer_scaled.columns]
    
    if not final_selected_features:
        print("Warning: Winning FS method found 0 features on Outer Fold. Scoring as 0.")
        outer_score = 0
    else:
        # Train the best Random Forest on the outer training set
        final_rf = RandomForestClassifier(**best_rf_params, bootstrap=True, max_features='sqrt', random_state=42, n_jobs=-1)
        final_rf.fit(X_train_outer_scaled[final_selected_features], y_train_outer)
        
        # Test on the held-out outer test set
        final_preds = final_rf.predict(X_test_outer_scaled[final_selected_features])
        outer_score = accuracy_score(y_test_outer, final_preds)
    
    outer_results.append({
        'fold': outer_fold + 1,
        'best_fs_method': best_fs_method,
        'best_rf_params': best_rf_params,
        'n_features_selected': len(final_selected_features),
        'test_accuracy': outer_score
    })

#%% ---------------- FINAL RESULTS ----------------
results_df = pd.DataFrame(outer_results)
print("\n=== Final Outer Loop Results ===")
print(results_df)

if not results_df.empty:
    print(f"\nAverage Test Accuracy: {results_df['test_accuracy'].mean():.3f} +/- {results_df['test_accuracy'].std():.3f}")