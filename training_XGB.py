#%%
import pandas as pd
import numpy as np
import pickle

import sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, ValidationCurveDisplay, LearningCurveDisplay
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

from load_data import load_data, split_pd
from preprocessing import remove_highly_correlated_features
from fs_mutualinformation import fs_mutualinformation
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from load_data import load_data, split_pd
from preprocessing import remove_highly_correlated_features
from fs_mutualinformation import fs_mutualinformation
from fs_lasso import fs_lasso


sklearn.set_config(transform_output="pandas")

#%% Custom feature selection transformers for pipeline integration

# Transformer to remove highly correlated features 
class CorrelationFilter(BaseEstimator, TransformerMixin): 
    #het aanmaken van een klasse die de BaseEstimator en TransformerMixin van sklearn erft, zodat deze gebruikt kan worden in een sklearn pipeline. 
    #Deze klasse is bedoeld om features te verwijderen die sterk gecorreleerd zijn met andere features, wat kan helpen bij het verminderen van multicollineariteit en het verbeteren van de prestaties van het model.
    def __init__(self, threshold=0.95): #hier wordt de init methode gedefinieerd, waarbij een drempelwaarde voor correlatie wordt ingesteld. Standaard is deze drempel 0.95.
        self.threshold = threshold
        self.selected_features_ = None

    def fit(self, X, y=None): #hier wordt de fit methode gedefinieerd. In deze methode worden de sterk gecorreleerde features verwijderd op basis van de drempelwaarde die in de init methode is ingesteld. De functie remove_highly_correlated_features wordt gebruikt om deze features te identificeren en te verwijderen. 
        #De wel geselecteerde features worden opgeslagen in self.selected_features_ voor gebruik in de transform methode.
        _, self.selected_features_ = remove_highly_correlated_features(
            X, correlation_threshold=self.threshold, show_details=False
        )
        return self

    def transform(self, X): #hier wordt de transform methode gedefinieerd, waarbij alleen de geselecteerde features worden behouden in de output. De input X wordt gefilterd op basis van de geselecteerde features die in de fit methode zijn bepaald.
        return X[self.selected_features_]

# Transformer to perform feature selection using MI
class MIFilter(BaseEstimator, TransformerMixin):
    #het aanmaken van een klasse die de BaseEstimator en TransformerMixin van sklearn erft, zodat deze gebruikt kan worden in een sklearn pipeline.
    #Deze klasse is bedoeld om feature selectie uit te voeren op basis van Mutual Information (MI)
    def __init__(self, num_features=None):
        self.num_features = num_features
        self.selected_features_ = None

    def fit(self, X, y):
        y_series = pd.Series(y, index=X.index)
        self.selected_features_ = fs_mutualinformation(X, y_series, self.num_features)[0]
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
    ('variance', VarianceThreshold(threshold=0.0)), #verwijderen van zero variance features 
    ('correlation', CorrelationFilter(threshold=0.95)), #verwijderen van highly correlated features 
    ('MI', MIFilter()),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
]) # dit is de pipeline die uiteindelijk opgeslagen wordt, deze bevat alle stappen van preprocessing en feature selection tot aan het model zelf. De hyperparameters van de MI-filter en XGBClassifier worden later getuned in de GridSearchCV.

# Hyperparameter grid for both RFE and MI
param_grid = {
    'xgb__n_estimators': [50, 100, 200],
    'xgb__max_depth': [3, 4, 5],
    'xgb__learning_rate': [0.01, 0.05, 0.1],
    'xgb__subsample': [0.6, 0.8, 1.0],
    'xgb__colsample_bytree': [0.6, 0.8, 1.0],

    'MI__num_features': [10, 15, 20]
}

# Define stratified k-fold cross-validation
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform grid search to tune the hyperparameters
grid_search = GridSearchCV( #grid search CV zorgt ervoor dat elke mogelijke combinatie geevalueerd wordt
    estimator=pipeline,
    param_grid=param_grid, #dit is de volledige parameter grid waarbij de ideale combinatie van hyperparameters gezocht moet worden. 
    cv=cv_strategy, #5-fold stratified cross-validation om de hyperparameters te tunen. 
    scoring={'accuracy': 'accuracy', 'roc_auc': 'roc_auc'},
    n_jobs=-1,
    verbose=1,
    refit='roc_auc'  
)

# Hyperparameter tuning
print("Start tuning...")
grid_search.fit(GIST_train, y_train_encoded) #hier vindt het daadwerkelijke training proces plaats, waarbij de pipeline wordt getraind op de trainingsdata voor elke combinatie van hyperparameters in de param_grid, 
# alle mogelijke combinaties worden doorgelopen en de auc score en accuracy worden ook meteen bepaald

best_index = grid_search.best_index_ #dit is de beste combinatie van de hyperparameters 

# Calculate the mean ROC-AUC
cv_auc_mean = grid_search.cv_results_['mean_test_roc_auc'][best_index] #hierbij wordt het gemiddelde van de auc scores bepaald over de 5-folds voor de beste combinatie aan hyperparameters
cv_auc_std = grid_search.cv_results_['std_test_roc_auc'][best_index]
cv_acc_mean = grid_search.cv_results_['mean_test_accuracy'][best_index]
cv_acc_std = grid_search.cv_results_['std_test_accuracy'][best_index]

print("\n=== CV RESULTS (VALIDATION SETS) ===")
print(f"Beste parameters: {grid_search.best_params_}")
print(f"CV ROC-AUC:    {cv_auc_mean:.4f} (+/- {cv_auc_std:.4f})")
print(f"CV Accuracy:   {cv_acc_mean:.4f} (+/- {cv_acc_std:.4f})")

best_final_model = grid_search.best_estimator_ #dit is het uiteindelijke model dat getraind is op de volledige trainingsset 

# Check for overfitting on training set
y_pred_train = best_final_model.predict(GIST_train) #voorspelt de klasse van de labels per sample in de trainingsset op basis van de getrainde pipeline met beste hyperparameters
y_proba_train = best_final_model.predict_proba(GIST_train)[:, 1] #hier wordt voorspelt wat de kans is dat elke sample in de trainigsset behoort tot de GIST groep

print("\n=== TRAIN RESULTS (OVERFITTING CHECK) ===")
print(f"Train Accuracy: {accuracy_score(y_train_encoded, y_pred_train):.4f}") #hier wordt de accuracy bepaald, dus het percentage van de correct geclassificeerde samples
print(f"Train AUC:      {roc_auc_score(y_train_encoded, y_proba_train):.4f}") #hier wordt de AUC bepaald, dus hoe goed het model onderscheid kan maken tussen de twee klassen (GIST vs non-GIST) op basis van de voorspelde kansen

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
param_name = "MI__num_features"
param_range = [5, 10, 15, 20, 25, 30]

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
# %% # %% Save pickle of tuned hyperparameters and scores

filename = 'models/final_model_MI_XGB.pkl'

with open(filename, 'wb') as file:
    pickle.dump(best_final_model, file) #in de pickle zit het uiteindelijke model dat getraind is op de volledige trainingsset met de beste hyperparameters, inclusief alle preprocessing stappen en feature selectie

print(f"\nModel succesvol opgeslagen als: {filename}")

# %%
