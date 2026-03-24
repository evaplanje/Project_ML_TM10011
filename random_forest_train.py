
#%%
import itertools
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from load_data import load_data, split_pd
from preprocessing import remove_zero_variance_features, remove_highly_correlated_features

from fs_lasso import fs_lasso
from fs_mRMR import fs_mrmr
from fs_mutualinformation import fs_mutualinformation
from fs_RFE import perform_rfe

# %%
# 1. Data inladen en basis voorbereiding
GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)

# Zero variance mag buiten de loop (het kijkt alleen naar de X-waarden, niet naar y)
preproc_GIST_train, _ = remove_zero_variance_features(GIST_train, show_details=False)

X_full = preproc_GIST_train 
y_full = y_train.values 

label_encoder = LabelEncoder()
y_full = label_encoder.fit_transform(y_full)

# Optioneel: prints je eerdere resultaten
try:
    results_cv = pd.read_csv('nested_cv_results_XGB.csv')
    print("Eerdere resultaten:")
    print(results_cv)
except FileNotFoundError:
    pass

# %%
# 2. Hyperparameter grid en instellingen
xgb_param_grid = {
    'n_estimators': [50, 100, 200],      # Aantal bomen
    'max_depth': [3, 4, 5],              # Ondiepe bomen tegen overfitting
    'learning_rate': [0.01, 0.05, 0.1],  # Stapgrootte
    'subsample': [0.6, 0.8, 1.0],        # Fractie samples per boom
    'colsample_bytree': [0.6, 0.8, 1.0]  # Fractie features per boom
}

base_classifier = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Cross-validation setup
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

auc_scores = []
accuracy_scores = []
best_parameters_per_fold = []

# %%
# 3. Geneste Cross-Validation Loop (ZONDER Data Leakage)
for fold, (train_index, val_index) in enumerate(outer_cv.split(X_full, y_full), 1):
    print(f"--- Start Fold {fold} ---")
    
    # A. Split data (NÓG GEEN feature selection gedaan)
    X_tr = X_full.iloc[train_index]
    X_val = X_full.iloc[val_index]
    y_tr = y_full[train_index]
    y_val = y_full[val_index]
    
    # B. Feature Selection: Correlatie (ALLEEN fitten op X_tr)
    X_tr_filtered, selected_features_corr = remove_highly_correlated_features(
        X_tr,
        correlation_threshold=0.95,
        show_details=False
    )
    # Pas de gevonden niet-gecorreleerde features toe op de validatieset
    X_val_filtered = X_val[selected_features_corr]
#//////////////////////////////////////////////////////////////
#_______________________________________________________________
#Kies hier een andere versie afhankelijk van je feature selectie methode
#_______________________________________________________________
   # C. Feature Selection: mRMR (ALLEEN fitten op X_tr en y_tr)
    # num_features_to_select = 15
    
    # # DE FIX: Maak van de numpy array (y_tr) een Pandas Series met de index van X_tr_filtered
    # y_tr_series = pd.Series(y_tr, index=X_tr_filtered.index)
    
    # # Gebruik nu y_tr_series voor mRMR in plaats van y_tr
    # # (Check even of jouw fs_mrmr functie direct een lijst teruggeeft of een tuple; 
    # # bij fs_mutualinformation gebruikte je nog [0] erachter, bij mrmr is dat vaak niet nodig)
    # selected_features_mrmr = fs_mrmr(X_tr_filtered, y_tr_series, num_features_to_select)[0]
    
    # # Pas de definitieve selectie toe op zowel train als validatie
    # X_tr_final = X_tr_filtered[selected_features_mrmr]
    # X_val_final = X_val_filtered[selected_features_mrmr]
#________________________________________________________________
    # C. Feature Selection: LASSO met vooraf getunede penalty score
    
    # y_tr_series = pd.Series(y_tr, index=X_tr_filtered.index)
    # # Vul hier jouw getunede penalty score in (bijv. alpha=0.05 of C=0.05, 
    # # afhankelijk van hoe jouw specifieke fs_lasso functie is geschreven)
    # getunede_penalty = 0.02

    # X_tr_final= fs_lasso(X_tr_filtered, y_tr_series, getunede_penalty)[0]
    
    # # Omdat lasso_output[0] waarschijnlijk getallen/coëfficiënten bevat, 
    # # maken we er een boolean mask van (True als het getal NIET 0 is)
    
    # # Roep fs_lasso aan met de penalty en pak [0] voor de lijst met namen
    # selected_features_lasso = X_tr_final.columns.tolist()
    
    # # Check even hoeveel features de LASSO met deze penalty heeft overgelaten
    # print(f"Aantal features geselecteerd door LASSO: {len(selected_features_lasso)}")
    
    # # Pas de definitieve selectie toe op zowel train als validatie
    # X_val_final = X_val_filtered[selected_features_lasso]
#_______________________________________________________________

## C. Feature Selection: RFE (Recursive Feature Elimination)
# C. Feature Selection: RFE (Recursive Feature Elimination)
    y_tr_series = pd.Series(y_tr, index=X_tr_filtered.index)
    aantal_features_rfe = 15  
    
    # DE FIX: We pakken [1] aan het einde, omdat daar jouw lijst met features zit!
    selected_features_rfe = perform_rfe(X_tr_filtered, y_tr_series, aantal_features_rfe)[1]
    
    print(f"Aantal features geselecteerd door RFE: {len(selected_features_rfe)}")
    
    # Pas de definitieve selectie toe op zowel train als validatie
    X_tr_final = X_tr_filtered[selected_features_rfe]
    X_val_final = X_val_filtered[selected_features_rfe]
#_______________________________________________________________
# C. Feature Selection: Mutual Information (ZONDER data leakage)
    # y_tr_series = pd.Series(y_tr, index=X_tr_filtered.index)
    
    # # Het aantal features dat je wilt selecteren (uit je eerdere pickles/instellingen)
    # aantal_features_mi = 15
    
    # # Roep fs_mutualinformation aan. We gebruiken weer [0] voor de lijst met namen!
    # selected_features_mi = fs_mutualinformation(X_tr_filtered, y_tr_series, aantal_features_mi, False)[0]
    
    # print(f"Aantal features geselecteerd door Mutual Information: {len(selected_features_mi)}")
    
    # # Pas de definitieve selectie toe op zowel train als validatie
    # X_tr_final = X_tr_filtered[selected_features_mi]
    # X_val_final = X_val_filtered[selected_features_mi]

#_______________________________________________________________

    # D. Hyperparameters tunen (Inner CV op X_tr_final)
    grid_search = GridSearchCV(
        estimator=base_classifier,
        param_grid=xgb_param_grid,
        cv=inner_cv,
        scoring='roc_auc',
        n_jobs=-1,  # Gebruik alle CPU cores
        verbose=0
    )
    
    grid_search.fit(X_tr_final, y_tr)
    
    best_model = grid_search.best_estimator_
    best_parameters_per_fold.append(grid_search.best_params_)
    print(f"Beste parameters: {grid_search.best_params_}")
    
    # E. Evaluatie op ongeziene data (Outer CV op X_val_final)
    y_val_pred_proba = best_model.predict_proba(X_val_final)[:, 1]
    auc = roc_auc_score(y_val, y_val_pred_proba)
    auc_scores.append(auc)
    
    print(f"AUC score voor fold {fold}: {auc:.4f}")


    y_val_pred_class = best_model.predict(X_val_final)
    accuracy = accuracy_score(y_val, y_val_pred_proba > 0.5)
    accuracy_scores.append(accuracy) # Voeg toe aan je nieuwe lijst!

    print(f"Accuracy score voor fold {fold}: {accuracy:.4f}\n")

# %%
# 4. Eindresultaten printen
print("=========================================")
print(f"Alle AUC Scores:      {[round(score, 4) for score in auc_scores]}")
print(f"Gemiddelde AUC:       {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")

print(f"Alle Accuracy Scores: {[round(score, 4) for score in accuracy_scores]}")
print(f"Gemiddelde Accuracy:  {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")
# %%
# --- Resultaten opslaan als Pickle ---

# 1. Verzamel alle lijstjes in één overzichtelijk Pandas DataFrame
results_df = pd.DataFrame({
    'fold': range(1, len(auc_scores) + 1),
    'best_xgb_params': best_parameters_per_fold,
    'roc_auc_score': auc_scores,
    'accuracy_score': accuracy_scores
})

# 2. Kies een duidelijke bestandsnaam
#PAS HIER NOG DE NAAM AAN ALS JE EEN ANDERE FEATURE SELECTIE METHODE GEBRUIKT 
pickle_filename = 'nested_cv_results_XGB_RFE.pkl'

# 3. Sla het op als pickle!
results_df.to_pickle(pickle_filename)

print(f"\n Resultaten succesvol opgeslagen in: {pickle_filename}")
print(results_df)

# Tip: Als je het later weer wilt inladen om te bekijken, gebruik je simpelweg:
# ingeladen_data = pd.read_pickle('nested_cv_results_XGB_RFE.pkl')
# %%
