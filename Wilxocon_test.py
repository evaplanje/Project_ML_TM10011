#%%
import pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from itertools import combinations

#%% # Load stored model performance scores for each classifier
with open('model_scores_RF.pkl', 'rb') as f:
    scores_RF = pickle.load(f)

with open('model_scores_SVM.pkl', 'rb') as f:
    scores_SVM = pickle.load(f)

with open('model_scores_XGB.pkl', 'rb') as f:
    scores_XGB = pickle.load(f)

# Combine the scores from all classifiers into a single dictionary
all_model_scores = {**scores_RF, **scores_SVM, **scores_XGB}

print("=== Models===")
for model, scores in all_model_scores.items():
    print(f"{model}: {scores}")

#%% Bonferroni correction
n_comparisons = len(list(combinations(all_model_scores.keys(), 2)))
alpha_corrected = 0.05 / n_comparisons

print(f"\nAantal vergelijkingen: {n_comparisons}")
print(f"Bonferroni gecorrigeerde alpha: {alpha_corrected:.5f}")

#%% Pairwise Wilcoxon tests
results = []

win_counts = {model: 0 for model in all_model_scores.keys()}  # Count for wins per model

# Perform pairwise comparisons between all model combinations
for (model_a, scores_a), (model_b, scores_b) in combinations(all_model_scores.items(), 2):
    
    differences = np.array(scores_a) - np.array(scores_b)
    
    if np.all(differences == 0):
        continue
    
    stat, p = wilcoxon(scores_a, scores_b)

    winner = model_a if np.mean(scores_a) > np.mean(scores_b) else model_b
    
    results.append({
        'model_a':     model_a,
        'model_b':     model_b,
        'mean_auc_a':  np.mean(scores_a),
        'mean_auc_b':  np.mean(scores_b),
        'p_value':     p,
        'significant': p < alpha_corrected,
        'winner':   winner   
    })

    win_counts[winner] += 1 # update win count for the winning model

results_df = pd.DataFrame(results)
print("\n=== All pairwise comparisons ===")
print(results_df.to_string(index=False))

#%% Counting the amount of wins per model
print("\n=== Numer of wins per model ===")

sorted_win_counts = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)

for model, count in sorted_win_counts:
    print(f"{model}: {count} wins")
#%% 