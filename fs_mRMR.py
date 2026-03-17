import pandas as pd
import numpy as np
from load_data import load_data, split_pd, explore_data, plot_feature_pairs, plot_heatmap
from preprocessing import apply_normalization, remove_zero_variance_features
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.datasets import make_classification
from preprocessing import remove_zero_variance_features

#%%
import pandas as pd
from mrmr import mrmr_classif

def fs_mrmr(X_train, y_train, K=10, show_details=True):
    """
    Voert mRMR feature selectie uit op de trainingsset en past
    dezelfde selectie toe op de testset.

    Parameters
    ----------
    X_train : pd.DataFrame
        Trainingsdata.
    y_train : pd.Series of array-like
        Trainingslabels.
    X_test : pd.DataFrame, optional
        Testdata waarop dezelfde featureselectie wordt toegepast.
    K : int, default=10
        Aantal te selecteren features.
    show_details : bool, default=True
        Print geselecteerde features.

    Returns
    -------
    selected_features : list
        Geselecteerde features.
    X_train_selected : pd.DataFrame
        Trainingsdata met geselecteerde features.
    X_test_selected : pd.DataFrame, optional
        Testdata met geselecteerde features.
    """

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)

    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)

    selected_features = mrmr_classif(X=X_train, y=y_train, K=K)

    X_train_selected = X_train[selected_features]

    if show_details:
        print(f"Geselecteerde {K} features met mRMR:")
        print(selected_features)

    return selected_features, X_train_selected


GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)
normalized_GIST_train, scaler = apply_normalization(GIST_train)
preproc_GIST_train, kept_features = remove_zero_variance_features(normalized_GIST_train, show_details=False)


selected_features_mrmr, X_train_mrmr = fs_mrmr(
    X_train=preproc_GIST_train,
    y_train=y_train,
    K=10,
    show_details=True
)
#%%