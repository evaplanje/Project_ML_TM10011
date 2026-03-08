#%%


import pandas as pd
import os
import numpy as np
from load_data import load_data, split_pd, explore_data, plot_feature_pairs, plot_heatmap
from preprocessing import apply_winsorization, apply_normalization
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel


#%%

def reduce_features(df, correlation_threshold=0.97, show_details = True):
    """
    removes features if:
        - variance  = 0 (all values are the same)
        - correlation > correlation_threshold (feature is very similar to another one) 
    Parameters
    ----------
    pd.DataFrame
    
    Returns
    -------
    pd.DataFrame
        reduced features dataframe
    """
    df_reduced_var = df.loc[:, (df != df.iloc[0]).any()]    

    corr_matrix = df_reduced_var.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    df_reduced_var_corr = df_reduced_var.drop(columns=to_drop)

    if show_details == True:
        print(f"Number of features before: {df.shape[1]}")
        print(f"Number of features remaining after var=0 drop: {df_reduced_var.shape[1]}")
        print(f"Number of features remaining after var=0 and corr=0.95 drop: {df_reduced_var_corr.shape[1]}")
    
    return df_reduced_var_corr


def lasso_feature_selection(X, y):
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X, y)

    selector = SelectFromModel(lasso, prefit=True)
    X_selected = selector.transform(X)

    selected_features = X.columns[selector.get_support()]

    print("Selected features:", len(selected_features))
    print(selected_features)

    X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

    return X_selected_df

#%%

GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, show_details = False)
winsorized_GIST_train = apply_winsorization(GIST_train)
normalized_GIST_train = apply_normalization(winsorized_GIST_train)
#%%

filtered_GIST_train = reduce_features(normalized_GIST_train, correlation_threshold=0.97, show_details = False)
feature_selected_GIST_train = lasso_feature_selection(filtered_GIST_train, y_train)