import pandas as pd
import numpy as np
from load_data import load_data, split_pd, explore_data, plot_feature_pairs, plot_heatmap
from preprocessing import apply_winsorization, apply_normalization
from sklearn.linear_model import LogisticRegressionCV
from sklearn import model_selection
from sklearn import metrics
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import svm
from sklearn import decomposition
from preprocessing import apply_winsorization, apply_normalization, remove_zero_variance_features
import seaborn
import matplotlib.pyplot as plt



def pca_feature_selection(df, n_components=1000, show_details = True):
    """
    PCA reduction filter
    """
    scaler = preprocessing.StandardScaler()
    X = df.values
    X_scaled = scaler.fit_transform(X)

    pca_model = decomposition.PCA(n_components=n_components)
    X_pca = pca_model.fit_transform(X_scaled)


    pca_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    df_reduced = pd.DataFrame(X_pca, columns=pca_columns, index=df.index)

    if show_details:
        print("Selected components:", X_pca.shape[1])
        print("Explained variance ratio:")
        print(pca_model.explained_variance_ratio_)
        print("Cumulative explained variance:")
        print(np.cumsum(pca_model.explained_variance_ratio_))

        seaborn.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y)

    return df_reduced, pca_model



GIST_data = load_data('GIST_radiomicFeatures.csv')
GIST_train, GIST_test, y_train, y_test = split_pd(GIST_data, show_details = False)
winsorized_GIST_train = apply_winsorization(GIST_train)
normalized_GIST_train = apply_normalization(winsorized_GIST_train)
preproc_GIST_train, kept_features = remove_zero_variance_features(normalized_GIST_train, show_details=False)

#%%
feature_selected_GIST_train, pca_model = pca_feature_selection(preproc_GIST_train, n_components=0.95, show_details=False)
print(feature_selected_GIST_train.head())


seaborn.scatterplot(
    x=feature_selected_GIST_train["PC2"],
    y=feature_selected_GIST_train["PC1"],
    hue=y_train
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA plot")
plt.show()