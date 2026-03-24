#%%
import itertools
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import roc_auc_score

from load_data import load_data, split_pd
from preprocessing import remove_zero_variance_features, remove_highly_correlated_features

from fs_lasso import fs_lasso
from fs_mRMR import fs_mrmr
from fs_mutualinformation import fs_mutualinformation
from fs_RFE import perform_rfe


# %%

GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)
preproc_GIST_train, _ = remove_zero_variance_features(GIST_train, show_details=False)

X = preproc_GIST_train 
y = y_train.values 

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

results_cv = pd.read_csv('nested_cv_results_XGB.csv')
print(results_cv)

# %%
# %%
# Jouw bestaande parameter grid
xgb_param_grid = {
    'n_estimators': [50, 100, 200],      # Number of trees
    'max_depth': [3, 4, 5],              # Keep trees shallow to prevent overfitting
    'learning_rate': [0.01, 0.05, 0.1],  # Step size shrinkage
    'subsample': [0.6, 0.8, 1.0],        # Fraction of samples used per tree
    'colsample_bytree': [0.6, 0.8, 1.0]  # Fraction of features used per tree
}

# Basis classifier opzetten (zonder de parameters die we gaan tunen)
base_classifier = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Outer CV voor performance evaluatie
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inner CV voor hyperparameter tuning
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

auc_scores = []
best_parameters_per_fold = []

# Geneste Cross-Validation Loop
for fold, (train_index, val_index) in enumerate(outer_cv.split(X, y), 1):
    print(f"--- Start Fold {fold} ---")
    
    # Split data voor deze outer fold
    X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
    y_tr, y_val = y[train_index], y[val_index]
    
    # GridSearchCV instellen met de inner CV
    grid_search = GridSearchCV(
        estimator=base_classifier,
        param_grid=xgb_param_grid,
        cv=inner_cv,
        scoring='roc_auc',
        n_jobs=-1,  # Gebruik -1 om alle beschikbare CPU cores in te zetten voor snelheid
        verbose=0
    )
    
    # 1. Hyperparameters TUNEN op de training data van deze fold
    grid_search.fit(X_tr, y_tr)
    
    # Sla de beste parameters van deze fold op
    best_model = grid_search.best_estimator_
    best_parameters_per_fold.append(grid_search.best_params_)
    print(f"Beste parameters: {grid_search.best_params_}")
    
    # 2. Performance TESTEN op de ongeziene validatie data
    y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_pred_proba)
    auc_scores.append(auc)
    print(f"AUC score voor fold {fold}: {auc:.4f}\n")

# %%
# Resultaten printen
print("=========================================")
print(f"Alle AUC Scores: {[round(score, 4) for score in auc_scores]}")
print(f"Gemiddelde AUC: {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")