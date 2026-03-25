#%%
import pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from itertools import combinations

#%% ---------------- LAAD ALLE PICKLE BESTANDEN ----------------
with open('model_scores_RF.pkl', 'rb') as f:
    scores_RF = pickle.load(f)

with open('model_scores_SVM.pkl', 'rb') as f:
    scores_SVM = pickle.load(f)

with open('model_scores_XGB.pkl', 'rb') as f:
    scores_XGB = pickle.load(f)

# Samenvoegen tot één dictionary
all_model_scores = {**scores_RF, **scores_SVM, **scores_XGB}

print("=== Geladen modellen ===")
for model, scores in all_model_scores.items():
    print(f"{model}: {scores}")

#%% ---------------- BONFERRONI CORRECTIE ----------------
n_comparisons = len(list(combinations(all_model_scores.keys(), 2)))
alpha_corrected = 0.05 / n_comparisons

print(f"\nAantal vergelijkingen: {n_comparisons}")
print(f"Bonferroni gecorrigeerde alpha: {alpha_corrected:.5f}")

#%% ---------------- PAARSGEWIJZE WILCOXON TESTS ----------------
results = []
win_counts = {model: 0 for model in all_model_scores.keys()}  # Teller voor wins per model

for (model_a, scores_a), (model_b, scores_b) in combinations(all_model_scores.items(), 2):
    
    differences = np.array(scores_a) - np.array(scores_b)
    
    # Wilcoxon vereist minimaal één niet-nul verschil
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

    win_counts[winner] += 1

results_df = pd.DataFrame(results)
print("\n=== Alle paarsgewijze vergelijkingen ===")
print(results_df.to_string(index=False))

# ---------------- WINNERS TELLEN ----------------
print("\n=== Aantal overwinningen per model ===")
for model, count in win_counts.items():
    print(f"{model}: {count} overwinningen")



#%% ------