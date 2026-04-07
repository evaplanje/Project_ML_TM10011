#%% Imports

import itertools
import pickle
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score

from load_data import load_data, split_pd
from preprocessing import remove_zero_variance_features, remove_highly_correlated_features

from fs_lasso import fs_lasso
from fs_mRMR import fs_mrmr
from fs_mutualinformation import fs_mutualinformation
from fs_RFE import perform_rfe

#%% Settings for FS and XGBoost

# Feature selection tuning grids
C_VALUES = [0.01, 0.02, 0.03] #dit is de inverse van de regularization strength voor LASSO, waarbij een kleinere waarde betekent dat er meer regularisatie is (meer features worden op 0 gezet)
K_VALUES = [10, 15, 20] #deze parameters gebruik je voor mRMR, MI en RFE, waarbij K het aantal features is dat geselecteerd wordt

# Feature Selection configurations
fs_configs = [{'method': 'lasso', 'param': c} for c in C_VALUES] + \
             [{'method': 'mrmr', 'param': k} for k in K_VALUES] + \
             [{'method': 'mi', 'param': k} for k in K_VALUES] + \
             [{'method': 'rfe', 'param': k} for k in K_VALUES]

# XGBoost hyperparameter grid
xgb_param_grid = {
    'n_estimators': [50, 100, 200],      # Number of trees
    'max_depth': [3, 4, 5],              # Keep trees shallow to prevent overfitting
    'learning_rate': [0.01, 0.05, 0.1],  # Step size shrinkage
    'subsample': [0.6, 0.8, 1.0],        # Fraction of samples used per tree
    'colsample_bytree': [0.6, 0.8, 1.0]  # Fraction of features used per tree
} #dit zijn de hyperparameters voor XGBoost die we gaan tunen in de inner loop van de Nested Cross-Validation, waarbij we verschillende combinaties van het aantal bomen, maximale diepte, learning rate, subsample ratio en colsample_bytree ratio zullen bekijken om de beste prestaties te vinden voor elke feature selection methode

# Create combinations
xgb_keys, xgb_values = zip(*xgb_param_grid.items())
xgb_param_combinations = [dict(zip(xgb_keys, v)) for v in itertools.product(*xgb_values)]
#hier worden alle mogelijke combinaties van de XGBoost hyperparameters gemaakt, zodat we deze kunnen gebruiken in de inner loop van de Nested Cross-Validation om te bepalen welke combinatie het beste werkt voor elke feature selection methode

#%% Load and prepare data 

GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, False)

X = GIST_train 
y = y_train.values 

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
#%% Perform the Nested Cross-Validation (NCV)

# Define outer and inner stratified K-fold cross-validation for NCV
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7) #er worden hier 5 folds gemaakt in de outer loop, waar de evaluatie van de combinaties plaatsvindt
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=7) #er worden hier 3 folds gemaakt in de inner loop, waar de hyperparameter tuning plaatsvindt voor elke feature selection methode
# de verhouding tussen de klassen blijft nu gelijk door middel van de stratified K-fold

outer_results = []

# Start the outer loop
for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    print(f"\n--- Starting Outer Fold {outer_fold + 1} ---")
    
    # Split the data into a train- and test(validation) set
    X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx] #het splitsen van de data in een training set (X_train_outer) en een test(validation) set (X_test_outer) voor de huidige fold van de outer loop, waarbij train_idx en test_idx de indices zijn die aangeven welke samples in de training en test sets moeten worden opgenomen
    y_train_outer, y_test_outer = y[train_idx], y[test_idx] #het splitsen van de target labels in een training set (y_train_outer) en een test(validation) set (y_test_outer) voor de huidige fold van de outer loop, waarbij train_idx en test_idx dezelfde indices gebruiken als bij het splitsen van de features, zodat de juiste labels worden gekoppeld aan de juiste samples in de training en test sets

    # Track the best configuration for each feature selection method
    best_results = {
        'lasso': {'score': -1, 'fs_config': None, 'xgb_params': None},
        'mrmr':  {'score': -1, 'fs_config': None, 'xgb_params': None},
        'mi':    {'score': -1, 'fs_config': None, 'xgb_params': None},
        'rfe':   {'score': -1, 'fs_config': None, 'xgb_params': None}
    }

    # Inner loop: Hyperparameter Tuning
    for fs_config in fs_configs: #de loop wordt afgegaan voor alle mogelijke parameters van de feature selection methods
        for xgb_params in xgb_param_combinations: #systematisch alle combinaties van feature-selection method en XGBoost intellingen doorlopen in de inner loop 
            
            inner_scores = []
            
            # Split the outer training set into inner train- and validation set
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer, y_train_outer): #hier wordt de op dat moment fold van de outer loop opnieuw gesplitst in een inner train en validatie set
                X_train_inner = X_train_outer.iloc[inner_train_idx] #de trainset wordt gebruikt om het mdodel de hyperparameters te laten leren en de feature selection toe te passen, zodat we kunnen bepalen welke features het belangrijkst zijn voor het voorspellen van de target labels (GIST or non-GIST) binnen deze fold van de inner loop
                X_val_inner = X_train_outer.iloc[inner_val_idx] #hier wordt de validatie set van de inner loop gemaakt, zodat we een subset van de data hebben die we kunnen gebruiken voor het evalueren van het model en het toepassen van feature selection inner loop

                # Normalisation using RobustScalar
                scaler_inner = RobustScaler()
                X_train_inner = pd.DataFrame(
                    scaler_inner.fit_transform(X_train_inner),
                    columns=X_train_inner.columns,
                    index=X_train_inner.index
                 )
                X_val_inner = pd.DataFrame(
                    scaler_inner.transform(X_val_inner), #dezelfde schaal van de normalisatie wordt toegepast op de validatieset
                    columns=X_val_inner.columns,
                    index=X_val_inner.index
                )

                y_train_inner = pd.Series(y_train_outer[inner_train_idx], index=X_train_inner.index) #hier worden de target labels voor de training set van de inner loop gemaakt, waarbij dezelfde indices worden gebruikt als bij het splitsen van de features, zodat de juiste labels worden gekoppeld aan de juiste samples in de training set van de inner loop
                y_val_inner = pd.Series(y_train_outer[inner_val_idx], index=X_val_inner.index) 

                # Remove zero variance features and highly correlated features
                X_train_inner, kept_var_features_inner = remove_zero_variance_features(X_train_inner, show_details=False)
                X_val_inner = X_val_inner[kept_var_features_inner] 

                X_train_inner, kept_features = remove_highly_correlated_features(
                    X_train_inner,
                    correlation_threshold=0.95,
                    show_details=False
                )

                X_val_inner = X_val_inner[kept_features] # Apply corr filter to val

                # Apply each feature selection method
                if fs_config['method'] == 'lasso':
                    _, selected_features = fs_lasso(X_train_inner, y_train_inner, C=fs_config['param'])
                elif fs_config['method'] == 'mrmr':
                    _, selected_features = fs_mrmr(X_train_inner, y_train_inner, K=fs_config['param'], show_details=False)
                elif fs_config['method'] == 'mi':
                    selected_features, _ = fs_mutualinformation(X_train_inner, y_train_inner, k=fs_config['param'], showdetails=False)
                elif fs_config['method'] == 'rfe':
                    _, selected_features = perform_rfe(X_train_inner, y_train_inner, n_features_to_select=fs_config['param'])
                #hier worden echt de feature selection methodes toegepast voor elke waarde van K of C. Hier wordt dus bepaald welke features er uiteindelijk overblijven
                
                selected_features = list(selected_features) 
                selected_features = [f for f in selected_features if f in X_train_inner.columns] #check om te controleren of niet perongeluk een zero-variance or highly correlated in de selected features is gebleven
                
                if not selected_features:
                    continue 
                
                # Keep only the selected features 
                X_train_inner_sel = X_train_inner[selected_features] #alleen de geselecteerde subset wordt voorgelegd aan de XGB om hyperparameters te tunen. Er wordt voor elke C of K een lijst gemaakt
                X_val_inner_sel = X_val_inner[selected_features]
                
                # Train and evaluate XGBoost on the inner fold
                xgb_model = XGBClassifier(**xgb_params, random_state = 7, n_jobs=-1, use_label_encoder=False, eval_metric='logloss') #hier wordt een XGBoost aangemaakt waarbij de hyperparameters worden ingesteld volgens de huidige combinatie uit de grid
                xgb_model.fit(X_train_inner_sel, y_train_inner) 
                # hier wordt het model getraind op de geselecteerde features en de labels van de inner trainingsset,
                # zodat deze specifieke combinatie van featureselectie en hyperparameters geëvalueerd kan worden op de inner validatieset
                # Elke combinatie aan fs parameters en XGBoost hyperparameters wordt dus getraind en geevalueerd in de inner loop
                
                probs = xgb_model.predict_proba(X_val_inner_sel)[:, 1] #de kans van elke sample in de validatieset op 1 (GIST) wordt geschat per combinatie van fs en XGB
                inner_scores.append(roc_auc_score(y_val_inner, probs)) #de auc score wordt bepaald voor deze specifieke combinatie van fs parameters en XGBoost hyperparameters. Deze wordt opgeslagen    

            if not inner_scores:
                continue
                
            # Calculate the average validation score of the inner loop
            avg_score = np.mean(inner_scores) #voor elke feature selection methode en XGBooost hyperparameters wordt de gemiddelde score bepaald over de drie inner loops.
            method = fs_config['method']

            # Update the dictionary with the best configuration for the feature selection methods -> hierin staan dus na de inner fold alleen nog de beste combinatie van de K of C. Elke feature selection methode komt dus hierin voor. Daarnaast wordt er per feature selection method aangeegeven welke parameters dan voor de XGBoost het beste zijn. 
            if avg_score > best_results[method]['score']: #alleen de best scorende combinatie wordt uiteindelijk in de best scores geplaatst
                best_results[method]['score'] = avg_score
                best_results[method]['fs_config'] = fs_config
                best_results[method]['xgb_params'] = xgb_params    
    
    for method, result in best_results.items():
        print(f"Best Configuration ({method.upper()}) -> FS: {result['fs_config']}, Score: {result['score']:.4f}, xgb Params: {result['xgb_params']}")

    # Outer loop: Evaluate the best model pipeline
    #je neemt hierbij de beste combinatie aan feature selection parameter met de daarbijhorende hyperparameters die in de innerloop gevonden is en past dit toe op de outer trainingset 
    
    # Normalisation using RobustScalar
    scaler_outer = RobustScaler()
    X_train_outer_scaled = pd.DataFrame(
        scaler_outer.fit_transform(X_train_outer),
        columns=X_train_outer.columns,
        index=X_train_outer.index
    )
    X_test_outer_scaled = pd.DataFrame(
        scaler_outer.transform(X_test_outer),
        columns=X_test_outer.columns,
        index=X_test_outer.index
    )

    # Remove zero variance features and highly correlated features
    X_train_outer_var, kept_var_features_outer = remove_zero_variance_features(X_train_outer_scaled, show_details=False)
    X_test_outer_filtered = X_test_outer_scaled[kept_var_features_outer] # Apply var filter to test

    X_train_outer_corr, kept_features_outer = remove_highly_correlated_features(
        X_train_outer_var,
        correlation_threshold=0.95,
        show_details=False
    )

    X_test_outer_corr = X_test_outer_filtered[kept_features_outer] # Apply corr filter to test
    y_train_outer_series = pd.Series(y_train_outer, index=X_train_outer_corr.index)

    # Train and evaluate the best configuration for each feature selection method
    for method, result in best_results.items():
        best_fs_config = result['fs_config'] #hier worden de best scorende fs parameter uit de inner loop gehaald met de daarbij horende combinatie van hyperparameters van de classifier
        best_xgb_params = result['xgb_params'] #hier worden de best scorende XGBoost hyperparameters uit de inner loop gehaald (per fs methode)

        if best_fs_config is None or best_xgb_params is None:
            continue

        # Apply the best feature selection method (met de al bepaalde parameter)
        if method == 'lasso':
            _, final_features = fs_lasso(X_train_outer_corr, y_train_outer_series, C=best_fs_config['param'])
        elif method == 'mrmr':
            _, final_features = fs_mrmr(X_train_outer_corr, y_train_outer_series, K=best_fs_config['param'], show_details=False)
        elif method == 'mi':
            final_features, _ = fs_mutualinformation(X_train_outer_corr, y_train_outer_series, k=best_fs_config['param'], showdetails=False)
        elif method == 'rfe':
            _, final_features = perform_rfe(X_train_outer_corr, y_train_outer_series, n_features_to_select=best_fs_config['param'])

        final_features = [f for f in final_features if f in X_train_outer_corr.columns]

        if not final_features:
            outer_score_auc = 0.5
            outer_score_acc = 0.0

        else:
            # Train the final XGBoost model and evaluate it on the outer test(validation) set
            final_xgb = XGBClassifier(**best_xgb_params,random_state = 7, n_jobs=-1, eval_metric='logloss') #hier wordt een XGBoost model gemaakt met de hyperparameters die het best scoorde in de inner loop (dit bestaat wel de bestecombinaties van hyperparameters)
            final_xgb.fit(X_train_outer_corr[final_features], y_train_outer) #hierbij wordt het model getraind, dit wordt dan het model dat als 'beste kandidaat' getest gaat worden met een validatie set

            probs = final_xgb.predict_proba(X_test_outer_corr[final_features])[:, 1]
            outer_score_auc = roc_auc_score(y_test_outer, probs) #per feature methode wordt er een AUC score bepaald 
            preds = (probs >= 0.5).astype(int)
            outer_score_acc = accuracy_score(y_test_outer, preds)

        outer_results.append({
            'fold': outer_fold + 1,
            'model_name': f"{method.upper()}_XGB", 
            'fs_method': method,
            'best_fs_param': best_fs_config['param'],
            'best_xgb_params': best_xgb_params,
            'n_features_selected': len(final_features),
            'roc_auc_score': outer_score_auc,
            'accuracy_score': outer_score_acc

        }) #je eindigt dan dus met 4 modellen aan het eind, waarbij er voor elk model 5 AUC en acc scores gegeven zijn

#%% Final results 
results_df = pd.DataFrame(outer_results)

print("\n" + "="*20 + " RESULTS " + "="*20)
print(results_df.to_string(index=False))

# Calculate and print the average and std per feature selection method
if not results_df.empty:
    print("\nAverage Test ROC AUC Score per Model:")
    summary_stats = results_df.groupby('model_name')['roc_auc_score'].agg(['mean', 'std'])
    for index, row in summary_stats.iterrows():
        print(f"{index}: {row['mean']:.3f} +/- {row['std']:.3f}")
    print("\nAverage Test Accuracy Score per Model:")
    summary_stats = results_df.groupby('model_name')['accuracy_score'].agg(['mean', 'std'])
    for index, row in summary_stats.iterrows():
        print(f"{index}: {row['mean']:.3f} +/- {row['std']:.3f}")

#%% Save pickle of the model scores
all_model_scores = {}

for _, row in results_df.iterrows():
    model_name = row['model_name']
    score = row['roc_auc_score']

    if model_name not in all_model_scores:
        all_model_scores[model_name] = []

    all_model_scores[model_name].append(score)

with open('models_scores_ncv/NCV_model_scores_XGB.pkl', 'wb') as f:
    pickle.dump(all_model_scores, f)

print("\nScores per model:")
for model, scores in all_model_scores.items():
    print(f"{model}: {[f'{s:.4f}' for s in scores]}")

print("\n=== Processing Complete ===")

#%%