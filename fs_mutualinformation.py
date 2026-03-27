#%% Imports

import pandas as pd
from sklearn.feature_selection import mutual_info_classif

#%% Definition feature selection MI

def fs_mutualinformation(df, labels, k, showdetails=False):
    
    """
    Performs feature selection using Mutual Information (MI) by ranking features
    based on their dependency with the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the input features.
    labels : pd.Series or array-like
        Target labels.
    k : int
        Number of top features to select.
    showdetails : bool, default=False
        Whether to print the top selected features.

    Returns
    -------
    selected_features_mi : list of str
        List of selected feature names based on MI ranking.
    mi_scores : np.ndarray
        Array containing the Mutual Information score for each feature.
"""
    # Perform Mutual Information feature scoring between features and labels
    mi_scores = mutual_info_classif(
        df, 
        labels, 
        discrete_features=False,
        random_state=7
    )

    # Create a new DataFrame with all features and the MI scores, sorted by importance
    mi_df = pd.DataFrame({
        'feature': df.columns,
        'mi_score': mi_scores
    }).sort_values(by='mi_score', ascending=False).reset_index(drop=True)

    # Select the top k features in the DataFrame
    selected_features_mi = mi_df['feature'].head(k).tolist()

    if showdetails:
        print("\nTop features based on Mutual Information:\n")
        print(mi_df.head(k).to_string(index=False))

    return selected_features_mi, mi_scores


