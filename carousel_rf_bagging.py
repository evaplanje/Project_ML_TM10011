#%%
import pandas as pd
import numpy as np
import itertools
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
# Let op: zorg dat je je eigen functies (zoals load_data, fs_lasso, etc.) hierboven importeert

#%%
# Feature selection methoden
fs_methods = {
    'lasso': fs_lasso,
    # 'groupwise': fs_groupwise,
    # 'pca': fs_pca,
    'statistical': fs_statistical#,
    #'rfe': perform_rfe
}

#%% Bagging Hyperparameters
# BaggingClassifier gebruikt standaard Decision Trees. We tunen het ensemble-proces.
bagging_param_grid = {
    'n_estimators': [10, 50, 100],      # Aantal bomen
    'max_samples': [0.5, 0.8, 1.0],     # Percentage van de rijen gebruikt per boom
    'max_features': [0.5, 0.8, 1.0]     # Percentage van de features gebruikt per boom
}

# Creëer een lijst van alle Bagging parameter combinaties
bagging_keys, bagging_values = zip(*bagging_param_grid.items())
bagging_param_combinations = [dict(zip(bagging_keys, v)) for v in itertools.product(*bagging_values)]

#%% Data inladen en voorbereiden
GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)
preproc_GIST_train, kept_features = remove_zero_variance_features(GIST_train, show_details=False)

X = preproc_GIST_train # Jouw pandas DataFrame (200, n_features)
y = y_train.values # Jouw classes array (200,)

# Outer loop: evalueert de ware performance van het model
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Inner loop: tunet de hyperparameters en feature selection
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

outer_results = []

for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    print(f"--- Starting Outer Fold {outer_fold + 1} ---")
    
    # Splits data voor de outer loop
    X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]
    
    # Schalen (altijd NA het splitsen om leakage te voorkomen)
    scaler = RobustScaler()
    X_train_outer_scaled = pd.DataFrame(scaler.fit_transform(X_train_outer), columns=X_train_outer.columns)
    X_test_outer_scaled = pd.DataFrame(scaler.transform(X_test_outer), columns=X_test_outer.columns)

    best_inner_score = -1
    best_fs_method = None
    best_bagging_params = None

    # --- INNER LOOP: Hyperparameter Tuning ---
    for fs_name, fs_func in fs_methods.items():
        for bagging_params in bagging_param_combinations:
            
            inner_scores = []
            
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer_scaled, y_train_outer):
                X_train_inner = X_train_outer_scaled.iloc[inner_train_idx]
                y_train_inner = y_train_outer[inner_train_idx]
                X_val_inner = X_train_outer_scaled.iloc[inner_val_idx]
                y_val_inner = y_train_outer[inner_val_idx]
                
                # 1. Pas Feature Selection toe op de Inner Training set
                _, selected_features = fs_func(X_train_inner, y_train_inner)
                
                # 2. Filter datasets zodat ze matchen met de geselecteerde features
                X_train_inner_sel = X_train_inner[selected_features]
                X_val_inner_sel = X_val_inner[selected_features]
                
                # 3. Train Bagging Classifier
                bagging_clf = BaggingClassifier(**bagging_params, random_state=42, n_jobs=-1)
                bagging_clf.fit(X_train_inner_sel, y_train_inner)
                
                # 4. Evalueer
                preds = bagging_clf.predict(X_val_inner_sel)
                score = roc_auc_score(y_val_inner, preds)
                inner_scores.append(score)
            
            # Gemiddelde score over de inner folds
            avg_inner_score = np.mean(inner_scores)
            
            # Sla de beste combinatie op
            if avg_inner_score > best_inner_score:
                best_inner_score = avg_inner_score
                best_fs_method = fs_name
                best_bagging_params = bagging_params

    print(f"Best Inner Tuning - FS: {best_fs_method}, Bagging Params: {best_bagging_params}")

    # --- OUTER LOOP: Evalueer de beste pipeline ---
    # Pas de winnende feature selection opnieuw toe op de HELE outer training set
    winning_fs_func = fs_methods[best_fs_method]
    _, final_selected_features = winning_fs_func(X_train_outer_scaled, y_train_outer)
    
    # Train de beste Bagging Classifier op de outer training set
    final_bagging = BaggingClassifier(**best_bagging_params, random_state=42, n_jobs=-1)
    final_bagging.fit(X_train_outer_scaled[final_selected_features], y_train_outer)
    
    # Test op de achtergehouden outer test set
    final_preds = final_bagging.predict(X_test_outer_scaled[final_selected_features])
    outer_score = roc_auc_score(y_test_outer, final_preds)
    
    outer_results.append({
        'fold': outer_fold + 1,
        'best_fs_method': best_fs_method,
        'best_bagging_params': best_bagging_params,
        'n_features_selected': len(final_selected_features),
        'roc_auc_score': outer_score
    })

# --- Final Results ---
results_df = pd.DataFrame(outer_results)
print("\n=== Final Outer Loop Results ===")
print(results_df)
print(f"\nAverage Test Accuracy: {results_df[''roc_auc_score''].mean():.3f} +/- {results_df[''roc_auc_score''].std():.3f}")

outer_results.append({
    'fold':               outer_fold + 1,
    'model_name':         f"{best_fs_method['method']}_Bag",
    'best_fs_param':      best_fs_method['param'],
    'best_rf_params':     best_bagging_params,
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