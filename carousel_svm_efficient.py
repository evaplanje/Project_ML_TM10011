#%%
import numpy as np
import pandas as pd
import itertools
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, roc_auc_score
from fs_lasso import fs_lasso
from load_data import load_data, split_pd
from preprocessing import remove_zero_variance_features
from fs_statistical import fs_statistical
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from fs_statistical import fs_statistical
from fs_RFE import perform_rfe
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline


# Suppress the specific scikit-learn warnings to keep the console clean
warnings.filterwarnings("ignore", message="l1_ratio parameter is only used when penalty is 'elasticnet'")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


#%% Setup Feature Selection and Hyperparameters
fs_methods = {
    'lasso': fs_lasso,
    'statistical': fs_statistical,
    'rfe': perform_rfe
}

SVM_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Create a list of all SVM parameter combinations
svm_keys, svm_values = zip(*SVM_param_grid.items())
svm_param_combinations = [dict(zip(svm_keys, v)) for v in itertools.product(*svm_values)]

#%% Load and Prepare Data
GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)
preproc_GIST_train, kept_features = remove_zero_variance_features(GIST_train, show_details=False)

X = preproc_GIST_train # Your pandas DataFrame (200, n_features)
y = y_train.values     # Your classes array (200,)

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
    
    # Scale data AFTER splitting to avoid leakage
    scaler = RobustScaler()
    X_train_outer_scaled = pd.DataFrame(scaler.fit_transform(X_train_outer), columns=X_train_outer.columns)
    X_test_outer_scaled = pd.DataFrame(scaler.transform(X_test_outer), columns=X_test_outer.columns)

    # Dictionary to keep track of scores for each (fs_method, svm_params) combination
    combination_scores = {}

    # --- OPTIMIZED INNER LOOP: Hyperparameter Tuning ---
    # 1. Iterate through the inner folds FIRST
    for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer_scaled, y_train_outer):
        X_train_inner = X_train_outer_scaled.iloc[inner_train_idx]
        y_train_inner = y_train_outer[inner_train_idx]
        X_val_inner = X_train_outer_scaled.iloc[inner_val_idx]
        y_val_inner = y_train_outer[inner_val_idx]
        
        # 2. Iterate through Feature Selection methods
        for fs_name, fs_func in fs_methods.items():
            
            # Apply Feature Selection EXACTLY ONCE per inner fold, per method
            _, selected_features = fs_func(X_train_inner, y_train_inner)
            
            # Filter training and validation sets to match selected features
            X_train_inner_sel = X_train_inner[selected_features]
            X_val_inner_sel = X_val_inner[selected_features]
            
            # 3. Iterate through SVM parameters
            for svm_params in svm_param_combinations:
                
                # Train SVM
                svm = SVC(**svm_params)
                svm.fit(X_train_inner_sel, y_train_inner)
                
                # Evaluate
                preds = svm.predict(X_val_inner_sel)
                score = accuracy_score(y_val_inner, preds)
                
                # Store the score in our dictionary using a tuple key
                param_key = tuple(svm_params.items())
                dict_key = (fs_name, param_key)
                
                if dict_key not in combination_scores:
                    combination_scores[dict_key] = []
                combination_scores[dict_key].append(score)

    # --- FIND THE WINNING COMBINATION ---
    best_inner_score = -1
    best_fs_method = None
    best_svm_params = None

    # Calculate the average score across the 3 inner folds for each combination
    for (fs_name, param_key), scores in combination_scores.items():
        avg_inner_score = np.mean(scores)
        
        if avg_inner_score > best_inner_score:
            best_inner_score = avg_inner_score
            best_fs_method = fs_name
            best_svm_params = dict(param_key) # Convert back to dictionary

    print(f"Best Inner Tuning - FS: {best_fs_method}, SVM Params: {best_svm_params}")

    # --- OUTER LOOP: Evaluate the Best Model Pipeline ---
    # Re-apply the winning feature selection to the ENTIRE outer training set
    winning_fs_func = fs_methods[best_fs_method]
    _, final_selected_features = winning_fs_func(X_train_outer_scaled, y_train_outer)
    
    # Train the best SVM on the outer training set
    final_svm = SVC(**best_svm_params, random_state=42)
    final_svm.fit(X_train_outer_scaled[final_selected_features], y_train_outer)
    
    # Test on the held-out outer test set
    final_preds = final_svm.predict(X_test_outer_scaled[final_selected_features])
    outer_score = accuracy_score(y_test_outer, final_preds)
    
    outer_results.append({
        'fold': outer_fold + 1,
        'best_fs_method': best_fs_method,
        'best_svm_params': best_svm_params,
        'n_features_selected': len(final_selected_features),
        'test_accuracy': outer_score
    })

# --- Final Results ---
results_df = pd.DataFrame(outer_results)
print("\n=== Final Outer Loop Results ===")
print(results_df)
print(f"\nAverage Test Accuracy: {results_df['test_accuracy'].mean():.3f} +/- {results_df['test_accuracy'].std():.3f}")