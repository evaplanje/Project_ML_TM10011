#%%


import pandas as pd
import os
import numpy as np
from load_data import load_data, split_pd, explore_data, plot_feature_pairs, plot_heatmap
from preprocessing import apply_winsorization, apply_normalization


#%%

def reduce_features(df, correlation_threshold=0.90):
    df = df.loc[:, (df != df.iloc[0]).any()]    

    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    df_reduced = df.drop(columns=to_drop)
    
    print(f"Dropped {len(to_drop)} redundant features.")
    print(f"Features remaining: {df_reduced.shape[1]}")
    
    return df_reduced

#%%

GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)
winsorized_GIST_train = apply_winsorization(GIST_train)
normalized_GIST_train = apply_normalization(winsorized_GIST_train)
#%%

filtered_GIST_train = reduce_features(normalized_GIST_train)
