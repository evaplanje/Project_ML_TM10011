#%% Import

import pandas as pd
import numpy as np
import pickle
import sklearn
import scipy.stats
import xgboost

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import roc_curve, auc

from load_data import load_data, split_pd
from preprocessing import remove_highly_correlated_features
from fs_mutualinformation import fs_mutualinformation
from fs_lasso import fs_lasso

sklearn.set_config(transform_output="pandas")
#%% #%% Custom feature selection transformers for pipeline integration

# Transformer to remove highly correlated features 
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.selected_features_ = None
    def fit(self, X, y=None):
        _, self.selected_features_ = remove_highly_correlated_features(X, correlation_threshold=self.threshold, show_details=False)
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

#%% #%% Implementation of DeLong's test for comparing ROC-AUC

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

#%% Load and prepare data 
GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)

label_encoder = LabelEncoder()
label_encoder.fit(y_train) 
y_test_encoded = label_encoder.transform(y_test)

#%% Model evaluations

# Define models to be evaluated 
models_to_test = [
    'models/final_model_LASSO_RF.pkl',
    'models/final_model_MI_RF.pkl',
    'models/final_model_MI_XGB.pkl'
]

model_scores = {}

# Evaluate each trained model on the testset
print("\n=== MODEL EVALUATION ===")
for model_file in models_to_test:
    try:
        with open(model_file, 'rb') as f:
            pipe = pickle.load(f)
            
        y_pred = pipe.predict(GIST_test)
        
        if hasattr(pipe, "predict_proba"):
            y_score = pipe.predict_proba(GIST_test)[:, 1]
        else:
            y_score = pipe.decision_function(GIST_test)
            
        # Store scores for DeLong
        model_scores[model_file] = y_score
        
        acc = accuracy_score(y_test_encoded, y_pred)
        auc = roc_auc_score(y_test_encoded, y_score)
        
        print(f"{model_file}:")
        print(f"   Accuracy: {acc:.4f} | AUC: {auc:.4f}")
        
        
    except FileNotFoundError:
        print(f"{model_file} not found. Check if the file is in your directory.")
    except Exception as e:
        print(f"Error loading {model_file}: {e}")

#%% DeLong's statistical comparison

print("\n=== DELONG'S TEST COMPARISONS (p-values) ===")

if len(model_scores) == 3:
    # LASSO_RF vs MI_RF
    p_lasso_rf_vs_mi_rf = delong_roc_test(
        y_test_encoded, 
        model_scores['models/final_model_LASSO_RF.pkl'], 
        model_scores['models/final_model_MI_RF.pkl']
    )
    print(f"LASSO_RF vs MI_RF: p-value = {p_lasso_rf_vs_mi_rf:.5f}")

    # 2. LASSO_RF vs MI_XGB
    p_lasso_rf_vs_mi_xgb = delong_roc_test(
        y_test_encoded, 
        model_scores['models/final_model_LASSO_RF.pkl'], 
        model_scores['models/final_model_MI_XGB.pkl']
    )
    print(f"LASSO_RF vs MI_XGB: p-value = {p_lasso_rf_vs_mi_xgb:.5f}")

    # 3. MI_RF vs MI_XGB
    p_mi_rf_vs_mi_xgb = delong_roc_test(
        y_test_encoded, 
        model_scores['models/final_model_MI_RF.pkl'], 
        model_scores['models/final_model_MI_XGB.pkl']
    )
    print(f"MI_RF vs MI_XGB: p-value = {p_mi_rf_vs_mi_xgb:.5f}")
    
else:
    print("Could not run DeLong's test because one or more models failed to load.")
#%% Plot ROC-AUC's in one figure

def plot_combined_roc(y_true, model_scores):
    plt.figure(figsize=(10, 8))
    
    # Define colors for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for (model_name, scores), color in zip(model_scores.items(), colors):
        # Calculate FPR and TPR
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        
        clean_name = model_name.replace('final_model_', '').replace('.pkl', '').upper() # clean names for legend
        
        plt.plot(fpr, tpr, color=color, lw=2, 
                 label=f'{clean_name} (AUC = {roc_auc:.3f})')

    # Plot the random chance line
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC curve comparison - test set (n=50)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.show()

if len(model_scores) > 0:
    plot_combined_roc(y_test_encoded, model_scores)
else:
    print("No model scores available to plot.")