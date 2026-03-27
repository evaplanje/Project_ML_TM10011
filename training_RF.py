#%% Import

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, ValidationCurveDisplay, LearningCurveDisplay
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, accuracy_score
from load_data import load_data, split_pd
from preprocessing import remove_highly_correlated_features
from fs_mutualinformation import fs_mutualinformation
from fs_lasso import fs_lasso
import sklearn

sklearn.set_config(transform_output="pandas")

#%% Custom feature selection transformers for pipeline integration

# Transformer to remove highly correlated features 
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.selected_features_ = None

    def fit(self, X, y=None):
        _, self.selected_features_ = remove_highly_correlated_features(
            X, correlation_threshold=self.threshold, show_details=False
        )
        return self

    def transform(self, X):
        return X[self.selected_features_]

# Transformer to perform feature selection using MI
class MIFilter(BaseEstimator, TransformerMixin):
    def __init__(self, num_features=10):
        self.num_features = num_features
        self.selected_features_ = None

    def fit(self, X, y):
        y_series = pd.Series(y, index=X.index)
        self.selected_features_ = fs_mutualinformation(X, y_series, self.num_features)[0]
        return self

    def transform(self, X):
        return X[self.selected_features_]

# Transformer to perform feature selection using LASSO 
class LASSOFilter(BaseEstimator, TransformerMixin):
    def __init__(self, C=0.01):
        self.C = C
        self.selected_features_ = None

    def fit(self, X, y):
        _, selected_features = fs_lasso(X, y, C=self.C)
        self.selected_features_ = selected_features
        return self

    def transform(self, X):
        return X[self.selected_features_]

#%% Load and prepare data 

GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

#%% Building pipeline 

pipeline = Pipeline([
    ('scaler', RobustScaler()), 
    ('variance', VarianceThreshold(threshold=0.0)), 
    ('correlation', CorrelationFilter(threshold=0.95)),
    ('MI', MIFilter()),
    ('LASSO', LASSOFilter()),
    ('rf', RandomForestClassifier(random_state=42))
])

# Hyperparameter grid for both RF and MI/LASSO

param_grid = {
    'rf__n_estimators': [100, 200, 300], 
    'rf__max_depth': [3, 5, 7],
    'rf__min_samples_split': [4, 6, 10], 
    'rf__min_samples_leaf': [2, 5, 10], 
    'rf__max_features': ['sqrt', 'log2', 0.3],
    'MI__num_features': [10, 15, 20], # select either MI or LASSO
    'LASSO__C': [0.01,0.02,0.03] # select either MI or LASSO 
}

# Define stratified k-fold cross-validation
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform grid search to tune the hyperparameters
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring={'accuracy': 'accuracy', 'roc_auc': 'roc_auc'},
    n_jobs=-1,
    verbose=1,
    refit='roc_auc' 
)

# Hyperparameter tuning
print("Start tuning...")
grid_search.fit(GIST_train, y_train_encoded)

best_index = grid_search.best_index_

# Calculate the mean ROC-AUC
cv_auc_mean = grid_search.cv_results_['mean_test_roc_auc'][best_index]
cv_auc_std = grid_search.cv_results_['std_test_roc_auc'][best_index]
cv_acc_mean = grid_search.cv_results_['mean_test_accuracy'][best_index]
cv_acc_std = grid_search.cv_results_['std_test_accuracy'][best_index]

print("\n=== CV RESULTs (VALIDATION SETS) ===")
print(f"Beste parameters: {grid_search.best_params_}")
print(f"CV ROC-AUC:    {cv_auc_mean:.4f} (+/- {cv_auc_std:.4f})")
print(f"CV Accuracy:   {cv_acc_mean:.4f} (+/- {cv_acc_std:.4f})")

best_final_model = grid_search.best_estimator_

# Check for overfitting on training set
y_pred_train = best_final_model.predict(GIST_train)
y_proba_train = best_final_model.predict_proba(GIST_train)[:, 1]

print("\n=== TRAIN RESULTS (OVERFITTING CHECK) ===")
print(f"Train Accuracy: {accuracy_score(y_train_encoded, y_pred_train):.4f}")
print(f"Train AUC:      {roc_auc_score(y_train_encoded, y_proba_train):.4f}")

#%% Learning curve
print("\nLearning Curve genereren...")
fig, ax = plt.subplots(figsize=(8, 5))
LearningCurveDisplay.from_estimator(
    best_final_model, GIST_train, y_train_encoded,
    cv=cv_strategy, scoring='roc_auc', n_jobs=-1, ax=ax,
    score_type="both", std_display_style="fill_between"
)
ax.set_title("Learning Curve (ROC-AUC)")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ["train", "val"], loc="best")
plt.show()

#%% Validation curve

print("\nValidation Curve genereren...")
param_name = "MI__num_features" # select either MI or LASSO
param_range = [5, 10, 15, 20, 25, 30]

param_name = "LASSO__num_features" # select either MI or LASSO
param_range = [0.1, 0.2, 0.3, 0.4, 0.5]

fig, ax = plt.subplots(figsize=(8, 5))
ValidationCurveDisplay.from_estimator(
    best_final_model, GIST_train, y_train_encoded,
    param_name=param_name, param_range=param_range,
    cv=cv_strategy, scoring="roc_auc", n_jobs=-1, ax=ax,
    score_type="both", std_display_style="fill_between"
)
ax.set_title(f"Validation Curve: {param_name}")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ["train", "val"], loc="best")
plt.show()
# %% Save pickle of tuned hyperparameters and scores

filename = 'models/final_model_MI_RF.pkl' # select either MI or LASSO
filename = 'models/final_model_LASSO_RF.pkl' # select either MI or LASSO

with open(filename, 'wb') as file:
    pickle.dump(best_final_model, file)

print(f"\nModel saved as: {filename}")
# %%
