#%%
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import sklearn
sklearn.set_config(transform_output="pandas")
from load_data import load_data, split_pd
from preprocessing import remove_highly_correlated_features
from fs_mRMR import fs_mrmr

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

class MRMRFilter(BaseEstimator, TransformerMixin):
    def __init__(self, num_features=15):
        self.num_features = num_features
        self.selected_features_ = None

    def fit(self, X, y):
        # MRMR heeft y nodig als Pandas Series met de juiste index
        y_series = pd.Series(y, index=X.index)
        self.selected_features_ = fs_mrmr(X, y_series, self.num_features)[0]
        return self

    def transform(self, X):
        return X[self.selected_features_]

# =====================================================================
# 2. Data inladen
# =====================================================================

GIST_data = load_data('GIST_radiomicFeatures.csv')
# Let op: we gebruiken nu zowel Train als Test!
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)

# Label encoding voor de targets (1x doen voor train en test)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# =====================================================================
# 3. Laad de opgeslagen Pipeline in
# =====================================================================
model_filename = 'final_pipeline_mrmr_xgb.pkl'

with open(model_filename, 'rb') as f:
    best_final_model = pickle.load(f)

print("Pipeline succesvol ingeladen!")
# =====================================================================
# 4. Evalueer het definitieve model op de onafhankelijke testset
# =====================================================================

y_test_pred_proba = best_final_model.predict_proba(GIST_test)[:, 1]
y_test_pred_class = best_final_model.predict(GIST_test)

final_auc = roc_auc_score(y_test_encoded, y_test_pred_proba)
final_accuracy = accuracy_score(y_test_encoded, y_test_pred_class)

print("\n=========================================")
print("  EINDRESULTATEN OP ONAFHANKELIJKE TESTSET")
print("=========================================")
print(f"Definitieve AUC:      {final_auc:.4f}")
print(f"Definitieve Accuracy: {final_accuracy:.4f}")
print("\nClassificatie Rapport:")
print(classification_report(y_test_encoded, y_test_pred_class, target_names=[str(c) for c in label_encoder.classes_]))
