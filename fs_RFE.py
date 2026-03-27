
#%% Imports

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

#%% Definition feature selection RFE

def perform_rfe(X_train, y_train, n_features_to_select=20):

    """
    Performs Recursive Feature Elimination (RFE) using Logistic Regression to select the most relevant features.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data containing the input features.
    y_train : pd.Series or array-like
        Training labels.
    n_features_to_select : int, default=20
        Number of features to select.

    Returns
    -------
    None
        No reduced DataFrame is returned.
    selected_features : list of str
        List of selected feature names.
    """
    # Initialize and apply RFE using Logistic Regression to select the most relevant features
    base_model = LogisticRegression(solver = 'liblinear',max_iter=10000, random_state=7)
    rfe = RFE(estimator=base_model, n_features_to_select=n_features_to_select, step=1) 
    rfe.fit(X_train, y_train)
    
    # Create a new list with the selected features by mRMR
    selected_features = X_train.columns[rfe.support_].tolist()

    return None, selected_features


