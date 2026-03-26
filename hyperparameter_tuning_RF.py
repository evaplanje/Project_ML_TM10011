#%%
import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

from load_data import load_data, split_pd
from preprocessing import remove_highly_correlated_features
from fs_mutualinformation import fs_mutualinformation
from fs_lasso import fs_lasso
import sklearn
sklearn.set_config(transform_output="pandas")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import ValidationCurveDisplay, LearningCurveDisplay

#%%
# =====================================================================
# 1. Custom Transformers voor in de Pipeline
# (Dit zorgt ervoor dat jouw functies naadloos samenwerken met sklearn)
# =====================================================================

class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.selected_features_ = None

    def fit(self, X, y=None):
        # Bereken gecorreleerde features en sla op welke we willen houden
        _, self.selected_features_ = remove_highly_correlated_features(
            X, correlation_threshold=self.threshold, show_details=False
        )
        return self

    def transform(self, X):
        # Filter de dataset
        return X[self.selected_features_]

class MIFilter(BaseEstimator, TransformerMixin):
    def __init__(self, num_features=None):
        self.num_features = num_features
        self.selected_features_ = None

    def fit(self, X, y):
        y_series = pd.Series(y, index=X.index)
        self.selected_features_ = fs_mutualinformation(X, y_series, self.num_features)[0]
        return self

    def transform(self, X):
        return X[self.selected_features_]

# class LASSOFilter(BaseEstimator, TransformerMixin):
#     def __init__(self, C=None):
#         self.C = C
#         self.selected_features_ = None

#     def fit(self, X, y):
#         _, selected_features = fs_lasso(X, y, C=self.C)
#         self.selected_features_ = selected_features
#         return self

#     def transform(self, X):
#         return X[self.selected_features_]
#%%
# =====================================================================
# 2. Data inladen
# =====================================================================

GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)

# Label encoding voor de targets (1x doen voor train en test)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
#%%
# =====================================================================
# 3. De Scikit-Learn Pipeline Bouwen (100% Data Leakage Vrij)
# =====================================================================

# Door 'set_output' te gebruiken, zorgen we dat Scikit-learn DataFrames 
# doorgeeft in plaats van Numpy arrays. Dat is nodig voor jouw functies!


pipeline = Pipeline([
    # Stap 1: Normalisatie (past zich per CV-fold aan)
    ('scaler', RobustScaler()), 
    
    # Stap 2: Zero variance (sklearn's native versie is makkelijker hier)
    ('variance', VarianceThreshold(threshold=0.0)), 
    
    # Stap 3: Jouw correlatiefilter
    ('correlation', CorrelationFilter(threshold=0.95)),
    
    # Stap 4: Jouw feature selectie
    ('MI', MIFilter()),
    #('LASSO', LASSOFilter()),
    
    # Stap 5: De classifier
    ('rf', RandomForestClassifier(random_state=42))
])
#%%
# =====================================================================
# 4. Hyperparameter Grid RANDOM FOREST
# =====================================================================

param_grid = {
    'rf__n_estimators': [100, 200, 300],       
    'rf__max_depth': [3, 5, 7],      
    'rf__min_samples_split': [4, 6, 10],     
    'rf__min_samples_leaf': [2, 5, 10],       
    'rf__max_features': ['sqrt', 'log2', 0.3],  

    'MI__num_features': [10, 15, 20]
    #'LASSO__C': [0.01, 0.02, 0.03]
}

# Cross-validatie setup voor het tunen op de train set
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv_strategy,
    # We berekenen nu beide:
    scoring={'accuracy': 'accuracy', 'auc_score': 'roc_auc'}, 
    n_jobs=-1,
    verbose=1,
    # Omdat we meerdere scores hebben, moeten we zeggen welke de doorslag geeft voor "best_params_"
    refit='auc_score'
)
#%%
# =====================================================================
# 5. Tunen en Fitten (Dit gebeurt uitsluitend op GIST_train)
# =====================================================================

print("Start met het tunen van het definitieve model op de trainingsdata...")
grid_search.fit(GIST_train, y_train_encoded)

print("\n=== TUNING RESULTATEN ===")
print(f"Beste parameters: {grid_search.best_params_}")

# Omdat we refit='auc_score' hebben gebruikt, is .best_score_ automatisch de AUC!
print(f"Beste Cross-Validation AUC: {grid_search.best_score_:.4f}")

# Haal de bijbehorende Accuracy op uit de resultaten
best_index = grid_search.best_index_
best_cv_accuracy = grid_search.cv_results_['mean_test_accuracy'][best_index]

print(f"Bijbehorende CV Accuracy: {best_cv_accuracy:.4f}")

# Haal het definitieve model op
best_final_model = grid_search.best_estimator_
print("Klassenvolgorde:", label_encoder.classes_)

# Evaluatie op train
y_pred = best_final_model.predict(GIST_train)
y_proba = best_final_model.predict_proba(GIST_train)[:, 1]

train_accuracy = accuracy_score(y_train_encoded, y_pred)
train_auc = roc_auc_score(y_train_encoded, y_proba)

tn, fp, fn, tp = confusion_matrix(y_train_encoded, y_pred).ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print("\n=== TESTRESULTATEN ===")
print(f"Accuracy:     {train_accuracy:.4f}")
print(f"AUC:          {train_auc:.4f}")
print(f"Sensitivity:  {sensitivity:.4f}")
print(f"Specificity:  {specificity:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_train_encoded, y_pred))
print(classification_report(
    y_train_encoded,
    y_pred,
    target_names=[str(c) for c in label_encoder.classes_]
))

# Sla het volledig getrainde eindmodel op
model_filename = 'final_pipeline_MI_rf.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(best_final_model, f)

print(f"\nDefinitieve pipeline opgeslagen als: {model_filename}")
# %%
#%%
# =====================================================================
# 6. Plotting the Learning Curve (Op GIST_train)
# =====================================================================

print("\nGenereren van de Learning Curve...")

# We gebruiken het beste model uit de grid search en jouw bestaande CV strategie
fig, ax = plt.subplots(figsize=(10, 6))

LearningCurveDisplay.from_estimator(
    estimator=grid_search.best_estimator_, 
    X=GIST_train, 
    y=y_train_encoded, 
    cv=cv_strategy,          # Gebruikt jouw StratifiedKFold van 5 splits
    scoring='accuracy',      # Zelfde metric als je GridSearchCV
    train_sizes=np.linspace(0.1, 1.0, 5), # 5 stappen van 10% tot 100% van de data
    n_jobs=-1,
    ax=ax,
    score_type="both",       # Plot zowel de train als de validation score
    std_display_style="fill_between" # Laat de variantie (schaduw) zien
)

ax.set_title("Learning Curve (Random Forest met LASSO)")
ax.set_xlabel("Aantal Training Samples")
ax.set_ylabel("Accuracy")
plt.grid(True)
plt.show()
#%%
# =====================================================================
# 7. Plotting the Validation Curve (Op GIST_train)
# =====================================================================

print("\nGenereren van de Validation Curve...")

# We pakken je beste model als basis, zodat de andere parameters 
# (zoals LASSO__C) op hun optimale waarde blijven staan.
base_estimator = grid_search.best_estimator_

# bij lasso
# param_name = "LASSO__C"
# param_range = np.arange(0.005, 0.04, 0.005) 

# bij MI
param_name = "MI__num_features"
param_range = np.arange(10, 40, 3) 


fig, ax = plt.subplots(figsize=(10, 6))

ValidationCurveDisplay.from_estimator(
    estimator=base_estimator,
    X=GIST_train,
    y=y_train_encoded,
    param_name=param_name,
    param_range=param_range,
    cv=cv_strategy,          # Gebruikt jouw StratifiedKFold van 5 splits
    scoring="auc_score",      # Zelfde metric als je GridSearchCV
    n_jobs=-1,
    ax=ax,
    score_type="both",       # Plot zowel train als validation
    std_display_style="fill_between" # Laat de variantie (schaduw) zien
)

ax.set_title(f"Validation Curve voor {param_name}")
ax.set_xlabel("Max Depth van Random Forest")
ax.set_ylabel("Accuracy")
# Omdat we een numerieke reeks gebruiken, zorgen we dat de x-as netjes verdeeld is
ax.set_xticks(param_range) 
plt.grid(True)
plt.show()

#%%