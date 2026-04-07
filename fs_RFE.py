
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
    base_model = LogisticRegression(solver = 'liblinear',max_iter=10000, random_state=7) #hier wordt een Logistic Regression model gebruikt als basis voor RFE, met de 'liblinear' solver die geschikt is voor kleinere datasets en L1 regularisatie, en een maximum van 10000 iteraties om convergentie te garanderen
    rfe = RFE(estimator=base_model, n_features_to_select=n_features_to_select, step=1) #hier wordt RFE geconfigureerd om het opgegeven aantal features te selecteren, waarbij in elke iteratie 1 feature wordt verwijderd (de minst belangrijke) totdat het gewenste aantal features is bereikt
    rfe.fit(X_train, y_train)  #hier wordt RFE toegepast op de trainingsdata, waarbij het model leert welke features het belangrijkst zijn voor het voorspellen van de target labels (GIST of non-GIST)
    
    # Create a new list with the selected features by RFE
    selected_features = X_train.columns[rfe.support_].tolist() #hier wordt een lijst gemaakt van de namen van de geselecteerde features, waarbij rfe.support_ een boolean array is die aangeeft welke features zijn geselecteerd (True) en welke niet (False)

    return None, selected_features


