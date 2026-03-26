#%%

import pandas as pd
import numpy as np
import pickle
import sklearn
import scipy.stats

from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, accuracy_score

# Import your custom functions
from load_data import load_data, split_pd
from preprocessing import remove_highly_correlated_features
from fs_mutualinformation import fs_mutualinformation
from fs_mRMR import fs_mrmr

sklearn.set_config(transform_output="pandas")

# =====================================================================
# 1. Custom Transformers (Required for Pickle)
# =====================================================================
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.selected_features_ = None
    def fit(self, X, y=None):
        _, self.selected_features_ = remove_highly_correlated_features(X, correlation_threshold=self.threshold, show_details=False)
        return self
    def transform(self, X):
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

class MRMRFilter(BaseEstimator, TransformerMixin):
    def __init__(self, num_features=15):
        self.num_features = num_features
        self.selected_features_ = None
    def fit(self, X, y):
        y_series = pd.Series(y, index=X.index)
        self.selected_features_ = fs_mrmr(X, y_series, self.num_features)[0]
        return self
    def transform(self, X):
        return X[self.selected_features_]

# =====================================================================
# 2. Fast DeLong's Test Implementation
# =====================================================================
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    tx = np.empty([positive_examples.shape[0], m], dtype=float)
    ty = np.empty([negative_examples.shape[0], n], dtype=float)
    tz = np.empty([predictions_sorted_transposed.shape[0], m + n], dtype=float)
    for r in range(predictions_sorted_transposed.shape[0]):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def calc_pvalue(aucs, sigma):
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return 10 ** (np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1)[0, 0])

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    order = np.lexsort((ground_truth,))
    ground_truth = ground_truth[order]
    predictions_one = predictions_one[order]
    predictions_two = predictions_two[order]
    label_1_count = np.sum(ground_truth == ground_truth[len(ground_truth) - 1])
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)

# =====================================================================
# 3. Load Data & Recreate Encoder
# =====================================================================
print("Loading data...")
GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)

label_encoder = LabelEncoder()
label_encoder.fit(y_train) 
y_test_encoded = label_encoder.transform(y_test)

# =====================================================================
# 4. Evaluation Loop
# =====================================================================
models_to_test = [
    'final_pipeline_MI_rf.pkl',
    'final_pipeline_mrmr_rf.pkl',
    'final_pipeline_mrmr_svm.pkl'
]

# Dictionary to store the continuous scores (probabilities/decision functions)
model_scores = {}

print("\n=== MODEL EVALUATION ===")
for model_file in models_to_test:
    try:
        with open(model_file, 'rb') as f:
            pipe = pickle.load(f)
            
        y_pred = pipe.predict(GIST_test)
        
        # Safe extraction of probabilities or decision scores for AUC/DeLong
        if hasattr(pipe, "predict_proba"):
            y_score = pipe.predict_proba(GIST_test)[:, 1]
        else:
            y_score = pipe.decision_function(GIST_test)
            
        # Store scores for DeLong later
        model_scores[model_file] = y_score
        
        acc = accuracy_score(y_test_encoded, y_pred)
        auc = roc_auc_score(y_test_encoded, y_score)
        
        print(f"{model_file}:")
        print(f"   Accuracy: {acc:.4f} | AUC: {auc:.4f}")
        
    except FileNotFoundError:
        print(f"{model_file} not found.")

# =====================================================================
# 5. DeLong Statistical Comparisons
# =====================================================================
print("\n=== DELONG'S TEST COMPARISONS (p-values) ===")

# Only run comparisons if all models loaded successfully
if len(model_scores) == 3:
    # 1. MI vs mRMR (Both Random Forest)
    p_mi_vs_mrmr = delong_roc_test(y_test_encoded, 
                                   model_scores['final_pipeline_MI_rf.pkl'], 
                                   model_scores['final_pipeline_mrmr_rf.pkl'])
    print(f"MI_RF vs mRMR_RF: p-value = {p_mi_vs_mrmr:.5f}")

    # 2. Random Forest vs SVM (Both mRMR)
    p_rf_vs_svm = delong_roc_test(y_test_encoded, 
                                  model_scores['final_pipeline_mrmr_rf.pkl'], 
                                  model_scores['final_pipeline_mrmr_svm.pkl'])
    print(f"mRMR_RF vs mRMR_SVM: p-value = {p_rf_vs_svm:.5f}")

    # 3. MI Random Forest vs mRMR SVM
    p_mi_rf_vs_svm = delong_roc_test(y_test_encoded, 
                                     model_scores['final_pipeline_MI_rf.pkl'], 
                                     model_scores['final_pipeline_mrmr_svm.pkl'])
    print(f"MI_RF vs mRMR_SVM: p-value = {p_mi_rf_vs_svm:.5f}")
    
else:
    print("Could not run DeLong's test because one or more models failed to load.")
