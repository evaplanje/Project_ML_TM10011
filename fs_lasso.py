
#%% Imports

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline


#%% Definition feature selection LASSO

def fs_lasso(
        df,
        y,
        C=0.1, 
        solver="saga", 
        max_iter=10000,
        class_weight="balanced",
        show_details=False):
    
    """
    Perform feature selection using standard regularized logistic regression (LASSO).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the input features
    y : pd.Series or array-like
        Target labels (GIST or non-GIST)
    C : float
        Inverse of the regularization strength
    solver : str
        Optimization algorithm used to fit the model
    max_iter : int
        Maximum number of iterations allowed for training
    class_weight : str or dict
        Weights associated with the classes to handle class imbalance 

    Returns
    -------
    df_selected : pd.DataFrame
        DataFrame containing only the selected features
    selected_features : list of str
        List of selected feature names
    """
    
    # Configure a standard LogisticRegression model using currently supported arguments
    model = LogisticRegression(
        penalty='l1',  #deze penalty is voor LASSO en det bepaalde coefficienten op 0 
        C=C,          
        solver=solver, 
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=7
    )
    
    # Build and fit the pipeline, retrieve the trained model, and extract the feature coefficients
    pipeline = Pipeline([("model", model)])
    pipeline.fit(df, y) #dit leert welke features belangrijk zijn en de coefficienten per feature
    fitted_model = pipeline.named_steps["model"] #haalt het getrainede model uit de pipeline
    
    if fitted_model.coef_.ndim > 1 and fitted_model.coef_.shape[0] == 1:
        coefs = fitted_model.coef_[0] # bij binary classificatie wordt alleen de eerste (en enige) rij gebruikt
    else:
        coefs = np.max(np.abs(fitted_model.coef_), axis=0) #als er meerdere klassen zijn, neem de maximale absolute waarde van de coefficients over alle klassen (in ons geval niet het geval)

    # Select features with non-zero coefficients
    selected_features = list(df.columns[coefs != 0]) #wanneer een feature een coefficient van niet 0 heeft, blijft de feature geselecteerd
    
    # Create a new DataFrame with the selected features by LASSO
    df_selected = df[selected_features]

    if show_details:
        print(f"Features before selection: {df.shape[1]}")
        print(f"Selected features: {len(selected_features)}")
        
        importance = pd.DataFrame({
            "feature": df.columns,
            "coef": coefs
        }) #hier wordt een tabel aangemaakt met de feature naam en coefficient erbij

        importance = importance[importance.coef != 0]
        importance["abs_coef"] = importance.coef.abs() #hier wordt een nieuwe kolom toegevoegd aan de tabel met de absolute waarde van de coefficienten, zodat we kunnen sorteren op belangrijkheid ongeacht de richting (positief of negatief)
        importance = importance.sort_values("abs_coef", ascending=False) #hier wordt de tabel gesorteerd op de absolute waarde van de coefficienten, zodat de meest belangrijke features bovenaan staan

        print("\nSelected features with LASSO:")
        print(importance[["feature", "coef"]].head(10))

    return df_selected, selected_features
