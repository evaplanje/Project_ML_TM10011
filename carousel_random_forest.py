import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import RobustScaler
import itertools
from fs_lasso import fs_lasso
from load_data import load_data, split_pd, explore_data, plot_feature_pairs, plot_heatmap
from preprocessing import apply_normalization, remove_zero_variance_features
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline


# from fs_groupwise import fs_groupwise
# from fs_pca import fs_pca
# from fs_statistical import fs_statistical



fs_methods = {
    'lasso': fs_lasso,

    # 'groupwise': fs_groupwise,
    # 'pca': fs_pca,
    # 'statistical': fs_statistical
}

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

# Create a list of all RF parameter combinations
rf_keys, rf_values = zip(*rf_param_grid.items())
rf_param_combinations = [dict(zip(rf_keys, v)) for v in itertools.product(*rf_values)]

# import
GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)
preproc_GIST_train, kept_features = remove_zero_variance_features(GIST_train, show_details=False)

X = preproc_GIST_train # Your pandas DataFrame (200, n_features)
y = y_train.values # Your classes array (200,)

# Outer loop evaluates the model's true performance
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Inner loop tunes the hyperparameters and feature selection
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

outer_results = []

for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    print(f"--- Starting Outer Fold {outer_fold + 1} ---")
    
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
                y_train_inner = y_train_outer[inner_train_idx]
                X_val_inner = X_train_outer_scaled.iloc[inner_val_idx]
                y_val_inner = y_train_outer[inner_val_idx]
                
                # 1. Apply Feature Selection on the Inner Training set
                _, selected_features = fs_func(X_train_inner, y_train_inner)
                
                # 2. Filter validation set to match selected features
                X_train_inner_sel = X_train_inner[selected_features]
                X_val_inner_sel = X_val_inner[selected_features]
                
                # 3. Train Random Forest
                rf = RandomForestClassifier(**rf_params, random_state=42)
                rf.fit(X_train_inner_sel, y_train_inner)
                
                # 4. Evaluate (e.g., using accuracy, or change to ROC-AUC)
                preds = rf.predict(X_val_inner_sel)
                score = accuracy_score(y_val_inner, preds)
                inner_scores.append(score)
            
            # Average score across inner folds for this combination
            avg_inner_score = np.mean(inner_scores)
            
            # Keep track of the best combination
            if avg_inner_score > best_inner_score:
                best_inner_score = avg_inner_score
                best_fs_method = fs_name
                best_rf_params = rf_params

    print(f"Best Inner Tuning - FS: {best_fs_method}, RF Params: {best_rf_params}")

    # --- OUTER LOOP: Evaluate the Best Model Pipeline ---
    # Re-apply the winning feature selection to the ENTIRE outer training set
    winning_fs_func = fs_methods[best_fs_method]
    _, final_selected_features = winning_fs_func(X_train_outer_scaled, y_train_outer)
    
    # Train the best Random Forest on the outer training set
    final_rf = RandomForestClassifier(**best_rf_params, random_state=42)
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

# --- Final Results ---
results_df = pd.DataFrame(outer_results)
print("\n=== Final Outer Loop Results ===")
print(results_df)
print(f"\nAverage Test Accuracy: {results_df['test_accuracy'].mean():.3f} +/- {results_df['test_accuracy'].std():.3f}")