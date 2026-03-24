#%%
import pandas as pd
import os
import numpy as np
from load_data import load_data, split_pd, explore_data, plot_feature_pairs, plot_heatmap
from preprocessing import apply_normalization, remove_zero_variance_features
from sklearn.feature_selection import mutual_info_classif

#%% Mutual information

def fs_mutualinformation(df, labels, k, showdetails=True):
    mi_scores = mutual_info_classif(
        df, 
        labels, 
        discrete_features=False,
        random_state=7
    )
    
    mi_df = pd.DataFrame({
        'feature': df.columns,
        'mi_score': mi_scores
    }).sort_values(by='mi_score', ascending=False).reset_index(drop=True)

    selected_features_mi = mi_df['feature'].head(k).tolist()

    if showdetails:
        print("\nTop features based on Mutual Information:\n")
        print(mi_df.head(k).to_string(index=False))

    return selected_features_mi, mi_scores


 #%% Inladen data
#  GIST_data = load_data('GIST_radiomicFeatures.csv')
# GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)
# normalized_GIST_train, scaler = apply_normalization(GIST_train)
# preproc_GIST_train, kept_features = remove_zero_variance_features(normalized_GIST_train, show_details=False)

# #%% Printen van de resultaten

# mi_results, _ = fs_mutualinformation(preproc_GIST_train, y_train, 20, False)


#%%
