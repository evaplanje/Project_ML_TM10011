#%%
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import seaborn as sns





def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, 'GIST_radiomicFeatures.csv'), index_col=0)

    return data

#%%
#kijken wat voor radiomic features er in de dataset zitten

print(load_data().head())
print(load_data().columns)
print(load_data().shape)
print(load_data()['label'].value_counts())

gist_data = load_data()

# %%
size=20
numeric_features = load_data().select_dtypes(include=['float64', 'int64']).iloc[:, :size]
corr_matrix = numeric_features.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title(f'Correlation Heatmap of the First {size} Features')
plt.show() 
# %%
