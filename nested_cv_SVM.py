#%% Imports
import itertools
import pickle
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, accuracy_score

from load_data import load_data, split_pd
from preprocessing import remove_zero_variance_features, remove_highly_correlated_features

from fs_lasso import fs_lasso
from fs_mRMR import fs_mrmr
from fs_mutualinformation import fs_mutualinformation
from fs_RFE import perform_rfe

#%% Settings for FS and SVM

# Feature selection tuning grids
C_VALUES = [0.01, 0.02, 0.03]
K_VALUES = [10, 15, 20]

# Feature Selection configurations
fs_configs = (
    [{'method': 'lasso', 'param': c} for c in C_VALUES] +
    [{'method': 'mrmr', 'param': k} for k in K_VALUES] +
    [{'method': 'mi', 'param': k} for k in K_VALUES] +
    [{'method': 'rfe', 'param': k} for k in K_VALUES]
)

# SVM hyperparameter grid
SVM_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Create combinations
svm_keys, svm_values = zip(*SVM_param_grid.items())
svm_param_combinations = [dict(zip(svm_keys, v)) for v in itertools.product(*svm_values)]

#%% Load and prepare data 

GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)

X = GIST_train
y = y_train.values

#%% Perform the Nested Cross-Validation (NCV)

# Define outer and inner stratified K-fold cross-validation for NCV
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)

outer_results = []

# Start the outer loop
for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    print(f"\n{'='*10} Outer Fold {outer_fold + 1} {'='*10}")

    # Split the data into a train- and test(validation) set
    X_train_outer = X.iloc[train_idx]
    X_test_outer = X.iloc[test_idx]
    y_train_outer = y[train_idx]
    y_test_outer = y[test_idx]

    # Track the best configuration for each feature selection method
    best_results = {
        'lasso': {'score': -1, 'fs_config': None, 'svm_params': None},
        'mrmr':  {'score': -1, 'fs_config': None, 'svm_params': None},
        'mi':    {'score': -1, 'fs_config': None, 'svm_params': None},
        'rfe':   {'score': -1, 'fs_config': None, 'svm_params': None}
    }

    # Inner loop: Hyperparameter Tuning
    for fs_config in fs_configs:
        for svm_params in svm_param_combinations:

            inner_scores = []

            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer, y_train_outer):

                # Split the outer training set into inner train- and validation set
                X_train_inner = X_train_outer.iloc[inner_train_idx].copy()
                X_val_inner = X_train_outer.iloc[inner_val_idx].copy()
                
                # Normalisation using RobustScalar
                scaler_inner = RobustScaler()
                X_train_inner = pd.DataFrame(
                    scaler_inner.fit_transform(X_train_inner),
                    columns=X_train_inner.columns,
                    index=X_train_inner.index
                )
                X_val_inner = pd.DataFrame(
                    scaler_inner.transform(X_val_inner),
                    columns=X_val_inner.columns,
                    index=X_val_inner.index
                )

                y_train_inner = pd.Series(y_train_outer[inner_train_idx], index=X_train_inner.index)
                y_val_inner = pd.Series(y_train_outer[inner_val_idx], index=X_val_inner.index)

                # Remove zero variance features and highly correlated features
                X_train_inner, kept_var_features_inner = remove_zero_variance_features(X_train_inner, show_details=False)
                X_val_inner = X_val_inner[kept_var_features_inner] 

                X_train_inner, kept_features = remove_highly_correlated_features(
                    X_train_inner,
                    correlation_threshold=0.95,
                    show_details=False
                )

                X_val_inner = X_val_inner[kept_features]

                # Apply the best feature selection method
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
                
                # Keep only the selected features 
                X_train_sel = X_train_inner[selected_features]
                X_val_sel = X_val_inner[selected_features]

                # Train and evaluate SVM on the inner fold
                svm = SVC(**svm_params, random_state= 7 ,probability=True)
                svm.fit(X_train_sel, y_train_inner)

                probs = svm.predict_proba(X_val_sel)[:, 1]
                inner_scores.append(roc_auc_score(y_val_inner, probs))

            if not inner_scores:
                continue
            
            # Calculate the average validation score of the inner loop
            avg_score = np.mean(inner_scores)
            method = fs_config['method']

            # Update the dictionary with the best configuration for the feature selection methods
            if avg_score > best_results[method]['score']:
                best_results[method]['score'] = avg_score
                best_results[method]['fs_config'] = fs_config
                best_results[method]['svm_params'] = svm_params    

    for method, result in best_results.items():
        print(f"Best Configuration ({method.upper()}) -> FS: {result['fs_config']}, Score: {result['score']:.4f}, SVM Params: {result['svm_params']}")

    # Outer loop: Evaluate the best model pipeline

    # Normalisation using RobustScalar
    scaler_outer = RobustScaler()
    X_train_outer_scaled = pd.DataFrame(
        scaler_outer.fit_transform(X_train_outer),
        columns=X_train_outer.columns,
        index=X_train_outer.index
    )
    X_test_outer_scaled = pd.DataFrame(
        scaler_outer.transform(X_test_outer),
        columns=X_test_outer.columns,
        index=X_test_outer.index
    )

    # Remove zero variance features and highly correlated features
    X_train_outer_var, kept_var_features_outer = remove_zero_variance_features(X_train_outer_scaled, show_details=False)
    X_test_outer_filtered = X_test_outer_scaled[kept_var_features_outer] # Apply var filter to test

    X_train_outer_corr, kept_features_outer = remove_highly_correlated_features(
        X_train_outer_var,
        correlation_threshold=0.95,
        show_details=False
    )

    X_test_outer_corr = X_test_outer_filtered[kept_features_outer] # Apply corr filter to test
    y_train_outer_series = pd.Series(y_train_outer, index=X_train_outer_corr.index)

    # Train and evaluate the best configuration for each feature selection method
    for method, result in best_results.items():
        best_fs_config = result['fs_config']
        best_svm_params = result['svm_params']

        if best_fs_config is None or best_svm_params is None:
            continue

        # Apply the best feature selection method
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
            outer_score_auc = 0.5
            outer_score_acc = 0

        else:
            # Train the final SVM model and evaluate it on the outer test(validation) set
            final_svm = SVC(**best_svm_params, random_state= 7,  probability=True)
            final_svm.fit(X_train_outer_corr[final_features], y_train_outer)

            probs = final_svm.predict_proba(X_test_outer_corr[final_features])[:, 1]
            outer_score_auc = roc_auc_score(y_test_outer, probs)
            preds = (probs >= 0.5).astype(int)
            outer_score_acc = accuracy_score(y_test_outer, preds)

        outer_results.append({
            'fold': outer_fold + 1,
            'model_name': f"{method.upper()}_SVM", 
            'fs_method': method,
            'best_fs_param': best_fs_config['param'],
            'best_svm_params': best_svm_params,
            'n_features_selected': len(final_features),
            'roc_auc_score': outer_score_auc,
            'accuracy_score': outer_score_acc


        })

#%% Final results 
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

#%% Save pickle of the model scores
all_model_scores = {}

for _, row in results_df.iterrows():
    model_name = row['model_name']
    score = row['roc_auc_score']

    if model_name not in all_model_scores:
        all_model_scores[model_name] = []

    all_model_scores[model_name].append(score)

with open('models_scores_ncv/NCV_model_scores_SVM.pkl', 'wb') as f:
    pickle.dump(all_model_scores, f)

print("\nScores per model:")
for model, scores in all_model_scores.items():
    print(f"{model}: {[f'{s:.4f}' for s in scores]}")

print("\n=== Processing Complete ===")

#%%