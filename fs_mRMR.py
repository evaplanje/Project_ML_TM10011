#%% Imports

import pandas as pd
from mrmr import mrmr_classif


#%% Definition feature selection mRMR

def fs_mrmr(X_train, y_train, K=10, show_details=False):
    """
    Performs mRMR feature selection on the training set and applies
    the same selection to the test set.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data.
    y_train : pd.Series or array-like
        Training labels.
    X_test : pd.DataFrame, optional
        Test data to which the same feature selection is applied.
    K : int, default=10
        Number of features to select.
    show_details : bool, default=True
        Whether to print the selected features.

    Returns
    -------
    selected_features : list
        Selected feature names.
    X_train_selected : pd.DataFrame
        Training data with selected features.
    X_test_selected : pd.DataFrame, optional
        Test data with selected features.
    """

    # Ensure that X_train and y_train are in pandas DataFrame and Series format
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)

    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)

    # Select features with the mrmr function
    selected_features = mrmr_classif(X=X_train, y=y_train, K=K, show_progress=False) #hier worden de K beste features geselecteerd door middel van minimum redundancy (weinig overlap met andere features) en maximum relevance (sterke correlatie met de target y_train)

    # Create a new DataFrame with the selected features by mRMR
    X_train_selected = X_train[selected_features] #houdt alleen de geselecteerde features over in de training set

    if show_details:
        print(f"Selected {K} features with mRMR:")
        print(selected_features)

    return selected_features, X_train_selected
