#%%
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
import itertools

from load_data import load_data, split_pd
from preprocessing import remove_zero_variance_features, remove_highly_correlated_features

from fs_lasso import fs_lasso
from fs_mRMR import fs_mrmr
from fs_mutualinformation import fs_mutualinformation
from fs_RFE import perform_rfe

#%% ---------------- SETTINGS ----------------

# Set your tuning grids here
C_VALUES = [0.01, 0.02, 0.03]
K_VALUES = [10, 15, 20]

# Build a list of all Feature Selection configurations we want to test
fs_configs = [{'method': 'lasso', 'param': c} for c in C_VALUES] + \
             [{'method': 'mrmr', 'param': k} for k in K_VALUES] + \
             [{'method': 'mi', 'param': k} for k in K_VALUES] + \
             [{'method': 'rfe', 'param': k} for k in K_VALUES]

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

#%% ---------------- LOAD & PREPROCESS ----------------

GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)
preproc_GIST_train, _ = remove_zero_variance_features(GIST_train, show_details=False)
preproc_GIST_train, _ = remove_highly_correlated_features(preproc_GIST_train, correlation_threshold=0.90, show_details=False)

X = preproc_GIST_train 
y = y_train.values 

#%% ---------------- NESTED CV ----------------

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

outer_results = []

for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    print(f"\n--- Starting Outer Fold {outer_fold + 1} ---")
    
    X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]
    
    scaler = RobustScaler()
    X_train_outer_scaled = pd.DataFrame(scaler.fit_transform(X_train_outer), columns=X_train_outer.columns)
    X_test_outer_scaled = pd.DataFrame(scaler.transform(X_test_outer), columns=X_test_outer.columns)

    best_inner_score = -1
    best_fs_config = None
    best_rf_params = None

    # --- INNER LOOP: Hyperparameter Tuning ---
    for fs_config in fs_configs:
        for rf_params in rf_param_combinations:
            
            inner_scores = []
            
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer_scaled, y_train_outer):
                X_train_inner = X_train_outer_scaled.iloc[inner_train_idx]
                X_val_inner = X_train_outer_scaled.iloc[inner_val_idx]

                y_train_inner = pd.Series(y_train_outer[inner_train_idx], index=X_train_inner.index)
                y_val_inner = pd.Series(y_train_outer[inner_val_idx], index=X_val_inner.index)
                
                # 1. Apply Feature Selection based on the current config
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
                
                rf = RandomForestClassifier(**rf_params, bootstrap=True, n_jobs=-1, random_state=42)
                rf.fit(X_train_inner_sel, y_train_inner)
                
                preds = rf.predict(X_val_inner_sel)
                inner_scores.append(roc_auc_score(y_val_inner, preds))
            
            if not inner_scores:
                continue
                
            avg_inner_score = np.mean(inner_scores)
            
            if avg_inner_score > best_inner_score:
                best_inner_score = avg_inner_score
                best_fs_config = fs_config
                best_rf_params = rf_params

    print(f"Best Inner Tuning - FS: {best_fs_config['method']} (param: {best_fs_config['param']}), RF: {best_rf_params}")

    # --- OUTER LOOP: Evaluate the Best Model Pipeline ---
    y_train_outer_series = pd.Series(y_train_outer, index=X_train_outer_scaled.index)
    
    # Re-apply the winning feature selection config
    if best_fs_config['method'] == 'lasso':
        _, final_selected_features = fs_lasso(X_train_outer_scaled, y_train_outer_series, C=best_fs_config['param'])
    elif best_fs_config['method'] == 'mrmr':
        _, final_selected_features = fs_mrmr(X_train_outer_scaled, y_train_outer_series, K=best_fs_config['param'], show_details=False)
    elif best_fs_config['method'] == 'mi':
        final_selected_features, _ = fs_mutualinformation(X_train_outer_scaled, y_train_outer_series, k=best_fs_config['param'], showdetails=False)
    elif fs_config['method'] == 'rfe':
        _, selected_features = perform_rfe(X_train_outer_scaled, y_train_outer_series, n_features_to_select=fs_config['param'])
    
    final_selected_features = list(final_selected_features)
    final_selected_features = [f for f in final_selected_features if f in X_train_outer_scaled.columns]
    
    if not final_selected_features:
        print("Warning: Winning FS method found 0 features on Outer Fold. Scoring as 0.")
        outer_score = 0
    else:
        final_rf = RandomForestClassifier(**best_rf_params, bootstrap=True, n_jobs=-1, random_state=42)
        final_rf.fit(X_train_outer_scaled[final_selected_features], y_train_outer)
        
        final_preds = final_rf.predict(X_test_outer_scaled[final_selected_features])
        outer_score = roc_auc_score(y_test_outer, final_preds)
    
    outer_results.append({
        'fold': outer_fold + 1,
        'best_fs_method': best_fs_config['method'],
        'best_fs_param': best_fs_config['param'],
        'best_rf_params': best_rf_params,
        'n_features_selected': len(final_selected_features),
        'roc_auc_score': outer_score
    })

#%% ---------------- FINAL RESULTS ----------------
results_df = pd.DataFrame(outer_results)
print("\n=== Final Outer Loop Results ===")
print(results_df)

if not results_df.empty:
    print(f"\nAverage Test Roc AUC score: {results_df['roc_auc_score'].mean():.3f} +/- {results_df['roc_auc_score'].std():.3f}")

outer_results.append({
    'fold':               outer_fold + 1,
    'model_name':         f"{best_fs_config['method']}_RF",
    'best_fs_param':      best_fs_config['param'],
    'best_rf_params':     best_rf_params,
    'n_features_selected': len(final_selected_features),
    'roc_auc_score':      outer_score
})


#%% ---------------- SAVE RESULTS ----------------
results_df = pd.DataFrame(outer_results)

# 1. CSV voor inspectie
results_df.to_csv('nested_cv_results_RF.csv', index=False)

# 2. Pickle voor Wilcoxon
all_model_scores = {}
for _, row in results_df.iterrows():
    model_name = row['model_name']
    if model_name not in all_model_scores:
        all_model_scores[model_name] = []
    all_model_scores[model_name].append(row['roc_auc_score'])

with open('model_scores_RF.pkl', 'wb') as f:
    pickle.dump(all_model_scores, f)

print("=== Opgeslagen ===")
print(results_df)
print(f"\nGemiddelde AUC: {results_df['roc_auc_score'].mean():.3f} +/- {results_df['roc_auc_score'].std():.3f}")
print(f"\nScores per model:\n{all_model_scores}")

#%%