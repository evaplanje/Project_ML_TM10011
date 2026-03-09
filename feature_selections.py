#%%

import pandas as pd
import os
import numpy as np
from preprocessing import GIST_train, low_variance_features, zero_variance_features

##% verwijderen van features met zero variance 
features_to_remove = set(zero_variance_features).union(set(low_variance_features))

GIST_train_clean = GIST_train.drop(columns=features_to_remove, errors='ignore')

#%% Normaliseren met IQR-methode

def normalise_iqr(X):
    X_normalised_iqr = X.copy()

    for feature in X.columns:
        Q1 = X[feature].quantile(0.25)
        Q3 = X[feature].quantile(0.75)
        IQR = Q3 - Q1
        if IQR != 0:
            X_normalised_iqr[feature] = (X[feature] - Q1) / IQR
        else:
            X_normalised_iqr[feature] = 0

    return X_normalised_iqr

X_normalised_iqr = normalise_iqr(GIST_train_clean)

#%% Correlatie filter

#%% SelectKbest features 

