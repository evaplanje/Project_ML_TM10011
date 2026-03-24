#%%
import itertools
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
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
# %%

xgb_param_grid = {
    'n_estimators': [50, 100, 200],      # Number of trees
    'max_depth': [3, 4, 5],              # Keep trees shallow to prevent overfitting
    'learning_rate': [0.01, 0.05, 0.1],  # Step size shrinkage
    'subsample': [0.6, 0.8, 1.0],        # Fraction of samples used per tree
    'colsample_bytree': [0.6, 0.8, 1.0]  # Fraction of features used per tree
}

final_selected_features_train = fs_mutualinformation(X, y, 20, False)[0] #verander dit naar features uit de pickles
#n_features_selected uit de pickles halen en hier gebruiken

GIST_train_train = GIST_train[final_selected_features_train]

GIST_train_train, final_selected_features_train= remove_highly_correlated_features(
                    GIST_train,
                    correlation_threshold=0.95,
                    show_details=False
                )


print(f"Final selected features: {final_selected_features_train}")
#%%

classifier = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
for train_index, val_index in skf.split(GIST_train, y):
    X_tr, X_val = GIST_train.iloc[train_index], GIST_train.iloc[val_index]
    y_tr, y_val = y[train_index], y[val_index]
    
    classifier.fit(X_tr, y_tr)
    y_val_pred_proba = classifier.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_pred_proba)
    auc_scores.append(auc)

print(f"AUC Scores: {auc_scores}")
print(f"Mean AUC: {np.mean(auc_scores)}")
# %%
