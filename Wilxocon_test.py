#%%
import pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt

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

sorted_win_counts = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)

for model, count in sorted_win_counts:
    print(f"{model}: {count} overwinningen")

#%% ---------------- PLOTTEN VAN DE AUC SCORES -----------------

# Maak een nieuwe dictionary met gemiddelde AUC voor elk model
mean_auc_scores = {model: np.mean(scores) for model, scores in all_model_scores.items()}
std_auc_scores = {model: np.std(scores) for model, scores in all_model_scores.items()}

# Matrix maken voor de heatmap
feature_selection_methods = ['LASSO', 'MRMR', 'MI', 'RFE']
classifiers = ['RF', 'SVM', 'XGB']
auc_matrix = pd.DataFrame(index=feature_selection_methods, columns=classifiers)
std_matrix = pd.DataFrame(index=feature_selection_methods, columns=classifiers)

# Vul de matrix met de gemiddelde AUC-scores
for fs_method in feature_selection_methods:
    for clf in classifiers:
        model_key = f'{fs_method}_{clf}' 
        auc_matrix.loc[fs_method, clf] = mean_auc_scores[model_key] 
        std_matrix.loc[fs_method, clf] = std_auc_scores[model_key]

auc_matrix = auc_matrix.apply(pd.to_numeric, errors='coerce') # nummerieke waardes maken 
std_matrix = std_matrix.apply(pd.to_numeric, errors='coerce')

# Maak de heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(auc_matrix, annot=True, cmap='Blues', cbar=True, fmt='.3f', linewidths=0.5)
# als je wel die std erbij wil, dan dit nog toevoegen annot_kws={'color': 'black'}

# Toevoegen std
# for (i, j), value in np.ndenumerate(auc_matrix.values):
#     std_value = std_matrix.iloc[i, j]
#     plt.text(j + 0.75, i + 0.5, f'±{std_value:.3f}', 
#              ha='center', va='center', color='black', fontsize=10)
    

plt.title('Mean AUC scores and std for each feature selection method and classifier')
plt.ylabel('Feature Selection Methoden')
plt.xlabel('Classifiers')
plt.show()
#%% ------